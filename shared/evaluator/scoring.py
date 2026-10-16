"""Task-configured quality gates, residency validation and final scoring."""
import json
import math
from pathlib import Path

from decode_benchmark import config, summarize
from inference_runtime import review_metadata, model_memory
from ppl_benchmark import validate_result as validate_ppl_result


def product_metrics(ppl, seconds):
    """Compute raw PPL * seconds before final_score applies the quality policy.

    reward = 1/(1 + C) is solely a bounded monotone encoding for Harbor's
    existing 0..1 result contract. It is not a Reference-relative score.
    """
    for name, value in [('ppl', ppl), ('decode_latency_seconds', seconds)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if ppl < 1:
        raise ValueError('PPL must be at least 1')
    product = ppl * seconds
    if not math.isfinite(product):
        raise ValueError('non-finite PPL-latency product')
    return {'ppl': ppl, 'decode_latency_seconds': seconds,
            'ppl_latency_product': product, 'reward': 1.0 / (1.0 + product),
            'reward_mapping': '1 / (1 + ppl * decode_latency_seconds)',
            'quality_gate_enabled': False, 'quality_pass': None}


def final_score(ppl_result, decode_result, *, decode_assets, ppl_assets, policy, public=False):
    cfg = config(decode_assets)
    model_memory.validate_report(ppl_result.get("model_memory"), policy)
    model_memory.validate_report(decode_result.get("model_memory"), policy)
    if ppl_result.get('valid') is not True:
        raise ValueError('no valid quality measurement')
    validate_ppl_result(ppl_result, ppl_assets, expected_split='public' if public else 'hidden')
    if decode_result.get('valid') is not True or decode_result.get('complete') is not True:
        raise ValueError('no valid complete decode measurement')
    if decode_result.get('measurement_protocol') != cfg['measurement_protocol']:
        raise ValueError('decode result uses the wrong protocol')
    summary = summarize(decode_result['requests'], cfg)
    gate = quality_gate(ppl_result['ppl'], ppl_assets)
    metrics = product_metrics(ppl_result['ppl'], summary['median_round_mean_seconds'])
    metrics.update(gate)
    score_cfg = json.loads((Path(ppl_assets) / 'ppl_config.json').read_text())
    _, apply_score = scoring_policy(score_cfg)
    apply_score(metrics, gate, ppl_assets)
    return {'valid': True, **review_metadata(), 'scoring_version': score_cfg['scoring_protocol'],
            'capacity_protocol': model_memory.PROTOCOL,
            'model_memory': decode_result.get('model_memory'),
            **{key:ppl_result[key] for key in ('ppl_protocol','ppl_split','predicted_tokens')},
            'public_only': public, 'is_official_score': not public, **summary,
            **metrics}


def _gate_bf16_penalty(ppl, cfg):
    """Same-split BF16: linear 1..1.5 cost penalty over 10..20%, inclusive cap."""
    baseline = cfg.get('bf16_ppl')
    if (cfg.get('baseline_status') != 'measured_new_protocol'
            or type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 1):
        raise ValueError('quality gate requires a measured task-owned BF16 PPL baseline')
    if type(ppl) not in (int, float) or not math.isfinite(ppl) or ppl < 1:
        raise ValueError('quality gate requires finite PPL >= 1')
    limit = baseline * 1.20
    start = baseline * 1.10
    if not math.isfinite(limit):
        raise ValueError('non-finite BF16-relative quality limit')
    passed = ppl <= limit
    # Direct threshold comparisons preserve the inclusive floating-point boundaries.
    # Rejected candidates report the capped multiplier too, but always earn zero.
    fraction = min(1.0, max(0.0, (ppl - start) / (limit - start)))
    multiplier = 1.0 + 0.5 * fraction
    return dict(quality_gate_enabled=True, quality_pass=passed, ranking_eligible=passed,
                quality_bf16_ppl=baseline, quality_max_ratio=1.20, quality_ppl_limit=limit,
                quality_gray_start_ratio=1.10, quality_gray_start_ppl=start,
                quality_max_penalty_multiplier=1.5, quality_penalty_multiplier=multiplier,
                ppl_increase_percent=100*(ppl/baseline-1),
                quality_rejection_reason=None if passed else 'ppl_exceeds_bf16_by_more_than_20_percent')


def _gate_bf16_cap(ppl, cfg):
    """Inclusive 30% quality cap relative to the BF16 baseline of this split."""
    baseline = cfg.get('bf16_ppl')
    if (cfg.get('baseline_status') != 'measured_new_protocol'
            or type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 1):
        raise ValueError('quality gate requires a measured task-owned BF16 PPL baseline')
    if cfg.get('quality_max_ratio') != 1.30:
        raise ValueError('this task requires a BF16-relative PPL cap of 1.30')
    if type(ppl) not in (int, float) or not math.isfinite(ppl) or ppl < 1:
        raise ValueError('quality gate requires finite PPL >= 1')
    limit = baseline * cfg['quality_max_ratio']
    if not math.isfinite(limit):
        raise ValueError('non-finite BF16-relative quality limit')
    passed = ppl <= limit
    return dict(quality_gate_enabled=True, quality_pass=passed, ranking_eligible=passed,
                quality_bf16_ppl=baseline, quality_max_ratio=1.30, quality_ppl_limit=limit,
                ppl_increase_percent=100*(ppl/baseline-1),
                quality_rejection_reason=None if passed else 'ppl_exceeds_bf16_by_more_than_30_percent')


def _gate_reference_limit(ppl, cfg):
    if (cfg['scoring_protocol'] in {
                'ppl-latency-gemma26-lowbit-quality-v1',
                'ppl-latency-gemma31-lowbit-quality-v1',
                'ppl-latency-qwen35-lowbit-quality-v1'}
            and cfg.get('reference_validation') != 'measured_and_frozen'):
        raise ValueError('reference calibration pending')
    baseline = cfg.get('bf16_ppl')
    ratio = cfg['quality_max_ratio']
    if (type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 1
            or cfg.get('baseline_status') != 'measured_new_protocol'):
        raise ValueError('quality gate requires a measured same-split BF16 baseline')
    if not math.isfinite(ppl) or ppl < 1:
        raise ValueError('invalid PPL')
    limit = cfg['quality_ppl_limit']
    if not math.isfinite(limit) or limit < 1 or limit != cfg['reference_ppl']:
        raise ValueError('invalid measured reference quality limit')
    passed = ppl <= limit
    return dict(quality_gate_enabled=True, quality_pass=passed, ranking_eligible=passed,
                quality_bf16_ppl=baseline, quality_max_ratio=ratio, quality_ppl_limit=limit,
                quality_gray_start_ratio=ratio, quality_gray_start_ppl=limit,
                quality_max_penalty_multiplier=1.0, quality_penalty_multiplier=1.0,
                ppl_increase_percent=100*(ppl/baseline-1),
                quality_rejection_reason=None if passed else 'ppl_exceeds_quality_limit')


def _score_bf16_penalty(metrics, gate, assets):
    adjusted = metrics['ppl_latency_product'] * gate['quality_penalty_multiplier']
    if not math.isfinite(adjusted):
        raise ValueError('non-finite penalized PPL-latency product')
    metrics['penalized_ppl_latency_product'] = adjusted
    metrics['reward'] = 1.0 / (1.0 + adjusted) if gate['quality_pass'] else 0.0
    metrics['reward_mapping'] = ('0 if ppl > 1.20 * split_bf16_ppl else '
                                 '1 / (1 + ppl * decode_latency_seconds * quality_penalty_multiplier)')


def _score_bf16_cap(metrics, gate, assets):
    if not gate['quality_pass']:
        metrics['reward'] = 0.0
    metrics['reward_mapping'] = ('0 if ppl > 1.30 * split_bf16_ppl else '
                                 '1 / (1 + ppl * decode_latency_seconds)')


def _score_reference_limit(metrics, gate, assets):
    reference = json.loads((Path(assets) / 'decode_config.json').read_text())
    passed = gate['quality_pass']
    metrics.update(speed_gate_enabled=False, speed_pass=None, speed_limit_seconds=None,
                   reference_decode_seconds=reference['reference_latency_seconds'],
                   target_met=passed, ranking_eligible=passed,
                   penalized_ppl_latency_product=metrics['ppl_latency_product'],
                   reward=1.0/(1.0+metrics['ppl_latency_product']) if passed else 0.0,
                   reward_mapping='0 unless quality gate passes; otherwise 1/(1+PPL*decode_seconds)')


# Each task chooses a policy explicitly; protocol names retain their original meaning.
SCORERS = {
    'bf16_penalty': (_gate_bf16_penalty, _score_bf16_penalty),
    'bf16_cap': (_gate_bf16_cap, _score_bf16_cap),
    'reference_limit': (_gate_reference_limit, _score_reference_limit),
}
_SCORING_POLICIES = {
    'ppl-latency-product-c4-bf16-gray15-cap20-v4': 'bf16_penalty',
    'ppl-latency-product-c4-bf16-cap30-v1': 'bf16_cap',
    'ppl-latency-gemma26-int4-quality-v1': 'reference_limit',
    'ppl-latency-gemma26-lowbit-quality-v1': 'reference_limit',
    'ppl-latency-gemma31-int4-quality-v1': 'reference_limit',
    'ppl-latency-gemma31-lowbit-quality-v1': 'reference_limit',
    'ppl-latency-qwen35-int4-quality-v1': 'reference_limit',
    'ppl-latency-qwen35-lowbit-quality-v1': 'reference_limit',
}


def scoring_policy(cfg):
    name = cfg.get('score_policy')
    if name not in SCORERS or name != _SCORING_POLICIES.get(cfg.get('scoring_protocol')):
        raise ValueError('task scoring policy does not match its pinned protocol')
    return SCORERS[name]


def quality_gate(ppl, assets):
    cfg = json.loads((Path(assets) / 'ppl_config.json').read_text())
    gate, _ = scoring_policy(cfg)
    return gate(ppl, cfg)
