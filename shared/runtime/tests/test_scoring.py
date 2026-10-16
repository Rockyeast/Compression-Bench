"""CPU checks for task-specific scoring boundaries and streamed decode integrity."""
import contextlib
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_task_tools import ROOT, TASKS, load, stage_checker


@contextlib.contextmanager
def modules(directory):
    runtime = load(directory / 'inference_runtime.py')
    ppl = load(directory / 'ppl_benchmark.py')
    with patch.dict(sys.modules, {'inference_runtime': runtime, 'ppl_benchmark': ppl}):
        decode = load(directory / 'decode_benchmark.py')
        with patch.dict(sys.modules, {'decode_benchmark': decode}):
            yield ppl, decode, load(directory / 'scoring.py')


def decode_rows(cfg):
    rows = []
    for r in range(cfg['rounds']):
        for p in range(cfg['requests_per_round']):
            times = [1 + i * (r + 1) / (cfg['output_tokens'] - 1) for i in range(cfg['output_tokens'])]
            rows.append(dict(round=r, request=p, arrival_seconds=times,
                chunk_arrival_seconds=times[:], chunk_sizes=[1] * len(times),
                input_tokens=cfg['input_tokens'], output_tokens=len(times),
                decode_intervals=len(times)-1, decode_seconds=times[-1]-times[0]))
    return rows


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def test_every_task_policy_and_inclusive_quality_boundary(self):
        for task in TASKS:
            target = self.work / task.name
            stage_checker(task, target)
            with modules(target) as (ppl, decode, scoring):
                for folder in ('environment/public-checker', 'tests'):
                    assets = task / folder / 'evaluator-assets'
                    cfg = json.loads((assets / 'ppl_config.json').read_text())
                    with self.subTest(task=task.name, split=cfg['split']):
                        policy = cfg['score_policy']
                        limit = (cfg['quality_ppl_limit'] if policy == 'reference_limit' else
                                 cfg['bf16_ppl'] * (1.2 if policy == 'bf16_penalty' else 1.3))
                        for value, passed in [(math.nextafter(limit, 0), True), (limit, True),
                                              (math.nextafter(limit, math.inf), False)]:
                            gate = scoring.quality_gate(value, assets)
                            self.assertEqual(gate['quality_pass'], passed)
                            metrics = scoring.product_metrics(value, 2.0)
                            scoring.scoring_policy(cfg)[1](metrics, gate, assets)
                            factor = gate.get('quality_penalty_multiplier', 1)
                            self.assertEqual(metrics['reward'], 1 / (1 + value * 2 * factor) if passed else 0)
                        if policy == 'bf16_penalty':
                            for ratio, multiplier in [(1.0, 1.0), (1.1, 1.0), (1.15, 1.25), (1.2, 1.5)]:
                                gate = scoring.quality_gate(cfg['bf16_ppl'] * ratio, assets)
                                self.assertAlmostEqual(gate['quality_penalty_multiplier'], multiplier)
                        for invalid in [0, -1, float('nan'), float('inf')]:
                            with self.assertRaises(ValueError):
                                scoring.quality_gate(invalid, assets)
                        for updates in [{'score_policy': 'unknown'}, {'scoring_protocol': 'unknown'}]:
                            with self.assertRaises(ValueError):
                                scoring.scoring_policy({**cfg, **updates})
                        if '-lowbit-' in cfg['scoring_protocol']:
                            with self.assertRaisesRegex(ValueError, 'calibration pending'):
                                scoring._gate_reference_limit(2.0, {**cfg, 'reference_validation': 'pending'})

    def test_decode_median_and_rejects_corrupt_streams(self):
        target = self.work / 'checker'
        stage_checker(TASKS[0], target)
        with modules(target) as (_, decode, _):
            cfg = decode.config(target / 'evaluator-assets')
            rows = decode_rows(cfg)
            self.assertEqual(decode.summarize(rows, cfg)['median_round_mean_seconds'], 2.0)
            cases = []
            cases.append(rows[:-1])
            cases.append([rows[0], *rows[1:-1], rows[0]])
            for field, value in [('chunk_sizes', [2] * (cfg['output_tokens']//2)),
                                 ('decode_seconds', 999), ('input_tokens', 0),
                                 ('arrival_seconds', [float('nan')] * cfg['output_tokens'])]:
                corrupt = copy.deepcopy(rows)
                corrupt[0][field] = value
                cases.append(corrupt)
            corrupt = copy.deepcopy(rows)
            corrupt[0]['arrival_seconds'][2] = 0
            corrupt[0]['chunk_arrival_seconds'][2] = 0
            cases.append(corrupt)
            for i, corrupt in enumerate(cases):
                with self.subTest(case=i), self.assertRaises(ValueError):
                    decode.summarize(corrupt, cfg)

    def test_raw_and_chat_ppl_exclude_only_protocol_prefix(self):
        # Gemma's 19-token prefix is covered by test_shared_evaluator; cover raw and Qwen chat here.
        for name, start in [('compress-qwen3.8-27b-8g', 1), ('compress-qwen3.6-35b-a3b-12g', 18)]:
            task = ROOT / 'tasks' / name
            target = self.work / name
            stage_checker(task, target)
            with modules(target) as (ppl, _, _):
                class Runtime:
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def token_logprobs(self, batch):
                        return [[None] + [-99.] * (start-1) + [-.5] * (2048-start) for _ in batch]
                    def memory_report(self): return None
                assets = target / 'evaluator-assets'
                result = ppl.measure(target, assets, assets / 'tokenizer', Runtime, expected_split='public')
                self.assertEqual(result['predicted_tokens'], 256 * (2048-start))
                self.assertAlmostEqual(result['ppl'], math.exp(.5))
                ppl.validate_result(result, assets, expected_split='public')
                with self.assertRaises(ValueError):
                    ppl.validate_result(result, assets, expected_split='hidden')


if __name__ == '__main__':
    unittest.main()
