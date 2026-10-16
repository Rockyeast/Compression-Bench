"""CPU regressions for shared entries; model stages are simulated, not GPU measurements."""
from contextlib import contextmanager, ExitStack
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

from test_task_tools import TASKS, load, stage_checker, stage_verifier
from test_scoring import decode_rows


@contextmanager
def entries(directory, public=False):
    with ExitStack() as stack:
        for name in ('inference_runtime', 'ppl_benchmark', 'submission_packaging', 'decode_benchmark', 'scoring', 'evaluate'):
            module = load(directory / (name + '.py'))
            stack.enter_context(patch.dict(sys.modules, {name: module}))
        yield module, load(directory / 'evaluate_public.py') if public else None


class EvaluationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_scoring_paths_preserve_all_task_results_and_capacity_checks(self):
        # 使用各任务真实配置和完整单 token 时间戳，验证公开入口与独立隐藏 score 命令。
        for task in TASKS:
            for public_mode in (True, False):
                with self.subTest(task=task.name, public=public_mode):
                    target = self.work / (task.name + ('-public' if public_mode else '-hidden'))
                    (stage_checker if public_mode else stage_verifier)(task, target)
                    assets = target / 'evaluator-assets'
                    with entries(target, public=public_mode) as (engine, public):
                        cfg = json.loads((assets / 'ppl_config.json').read_text())
                        dc = json.loads((assets / 'decode_config.json').read_text())
                        protocol = engine.model_memory.PROTOCOL
                        memory = dict(protocol=protocol, snapshots=[
                            dict(protocol=protocol, phase=phase,
                                 workers=[dict(bytes=16, storages=[dict(bytes=16)])])
                            for phase in ('load', 'post-inference')])
                        quality = dict(valid=True, ppl=cfg['bf16_ppl'], ppl_protocol=cfg['protocol'],
                                       ppl_split=cfg['split'], predicted_tokens=cfg['scored_tokens'], model_memory=memory)
                        speed = dict(valid=True, complete=True, measurement_protocol=dc['measurement_protocol'],
                                     requests=decode_rows(dc), model_memory=memory)
                        profile = dict(decode_assets=assets, ppl_assets=assets)
                        result = engine.score_measurements(quality, speed, profile, public_mode)
                        self.assertTrue(result['quality_pass'])
                        self.assertAlmostEqual(result['reward'], 1 / (1 + cfg['bf16_ppl'] * 2))
                        self.assertEqual(result['scoring_version'], cfg['scoring_protocol'])
                        with self.assertRaisesRegex(ValueError, 'resident memory exceeds limit'):
                            engine.final_score(quality, speed, decode_assets=assets, ppl_assets=assets,
                                               policy={'limit_gib': 1 / 2**30}, public=public_mode)
                        if not public_mode:
                            (target / 'ppl.json').write_text(json.dumps(quality))
                            (target / 'decode.json').write_text(json.dumps(speed))
                            output = target / 'score.json'
                            argv = ['evaluate.py', 'score', '--ppl-result', str(target / 'ppl.json'),
                                    '--decode-result', str(target / 'decode.json'), '--output', str(output)]
                            with patch.object(sys, 'argv', argv): engine.main()
                            self.assertEqual(json.loads(output.read_text()), result)


    def test_order_stop_on_failure_and_hidden_reward(self):
        for task in TASKS:
            target = self.work / task.name
            stage_verifier(task, target)
            assets = target / 'evaluator-assets'
            with entries(target) as (engine, _), self.subTest(task=task.name):
                for failure in (None, 'check', 'smoke', 'ppl', 'decode', 'score'):
                    with self.subTest(failure=failure):
                        calls = []

                        def operation(name, result):
                            def call(*args, **kwargs):
                                calls.append(name)
                                if name == failure:
                                    raise ValueError('simulated ' + name + ' failure')
                                return copy.deepcopy(result)
                            return call

                        @contextmanager
                        def package(*args):
                            calls.append('open')
                            try:
                                yield Path('submission')
                            finally:
                                calls.append('close')

                        output = target / ('logs-' + (failure or 'success'))
                        args = types.SimpleNamespace(submission=Path('submission'), assets=assets,
                            output_dir=output, limit_gib=engine.DEFAULT_LIMIT_GIB)
                        with patch.object(engine, 'inspect_submission', operation('check', {'valid': True})), \
                             patch.object(engine, 'evaluation_package', package), \
                             patch.object(engine, 'generate_smoke', operation('smoke', 'text')), \
                             patch.object(engine, 'measure_quality', operation('ppl', {'valid': True})), \
                             patch.object(engine, 'measure_decode', operation('decode', {'valid': True})), \
                             patch.object(engine, 'score_measurements', operation('score', {'valid': True, 'reward': .25})):
                            engine.run_verification(args)
                        stages = ['check', 'smoke', 'ppl', 'decode', 'score']
                        expected = stages if failure is None else stages[:stages.index(failure) + 1]
                        self.assertEqual([name for name in calls if name in stages], expected)
                        self.assertEqual(calls.count('open'), 0 if failure == 'check' else 1)
                        self.assertEqual(calls.count('close'), calls.count('open'))
                        self.assertEqual(float((output / 'reward.txt').read_text()), .25 if failure is None else 0)
                        result = json.loads((output / 'score.json').read_text())
                        self.assertEqual(result['valid'], failure is None)
                        for name in expected:
                            self.assertTrue((output / (name + '.json')).is_file())
                        for name in set(stages) - set(expected) - {'score'}:
                            self.assertFalse((output / (name + '.json')).exists())

    def test_reference_preflight_and_debug_adapter(self):
        for task in TASKS:
            target = self.work / task.name
            stage_checker(task, target)
            with entries(target, public=True) as (engine, public), self.subTest(task=task.name):
                assets = target / 'evaluator-assets'
                original = json.loads((assets / 'ppl_config.json').read_text())
                policy = engine.REFERENCE_PREFLIGHT
                for change in ({'reference_validation': 'pending', 'baseline_status': 'pending'}, {'bf16_ppl': None}):
                    (assets / 'ppl_config.json').write_text(json.dumps({**original, **change}))
                    blocked = policy != 'none' if 'baseline_status' in change else policy == 'bf16_baseline'
                    with patch.object(engine, 'inspect_submission', return_value={'valid': True}) as check, \
                         patch.object(engine, 'evaluation_package', side_effect=RuntimeError('reached model loading')):
                        with self.assertRaises(engine.SubmissionError if blocked else RuntimeError):
                            engine.evaluate_submission(Path('submission'), public.DEFAULT_PROFILE, public=True)
                        self.assertEqual(check.called, not blocked)
                        # File-only inspection remains usable while references are pending.
                        self.assertEqual(engine.evaluate_submission(Path('submission'), public.DEFAULT_PROFILE,
                                         public=True, package_only=True), {'check': {'valid': True}})
                (assets / 'ppl_config.json').write_text(json.dumps(original))
                if 'debug-9b' not in json.loads(public.PROFILE_CONFIG.read_text()):
                    continue
                profile = public.load_profile('debug-9b')
                seen_limits = []
                def measured(*args, **kwargs):
                    seen_limits.append(engine.model_memory.read_policy(target)['limit_gib'])
                    return dict(valid=True, ppl=2.0, ppl_protocol='raw', ppl_split='debug-9b', predicted_tokens=3)
                with patch.object(engine, 'measure_ppl', side_effect=measured) as main_measure, \
                     patch.object(engine, 'inspect_submission', return_value={'valid': True}), \
                     patch.object(engine, 'evaluation_package') as package, \
                     patch.object(engine, 'generate_smoke', return_value='text'), \
                     patch.object(engine, 'measure_decode', return_value={'valid': True}), \
                     patch.object(engine, 'score_measurements') as score:
                    package.return_value.__enter__.return_value = Path('submission')
                    engine.evaluate_submission(Path('submission'), profile, public=True)
                    self.assertEqual(seen_limits, [32])
                    main_measure.assert_called_once()
                    score.assert_not_called()
                self.assertEqual(engine.model_memory.read_policy(target)['limit_gib'], engine.DEFAULT_LIMIT_GIB)

    def test_public_and_hidden_report_fields(self):
        common = ('scoring_version', 'decode_latency_seconds', 'ppl_latency_product', 'reward_mapping',
                  'ranking_eligible', 'quality_bf16_ppl', 'quality_max_ratio', 'quality_ppl_limit',
                  'ppl_increase_percent', 'quality_rejection_reason')
        penalty = ('quality_gray_start_ratio', 'quality_gray_start_ppl', 'quality_max_penalty_multiplier',
                   'quality_penalty_multiplier', 'penalized_ppl_latency_product')
        speed = ('target_met', 'speed_pass', 'speed_limit_seconds', 'reference_decode_seconds')
        for task in TASKS:
            target = self.work / task.name
            stage_checker(task, target)
            with entries(target, public=True) as (engine, public), self.subTest(task=task.name):
                score = {k: 1 for k in (*common, *penalty, *speed)}
                score.update(reward=.25, quality_gate_enabled=True, quality_pass=True)
                quality = dict(ppl=2.0, ppl_protocol='fixture', ppl_split='public', predicted_tokens=3)
                measured = dict(check={'valid': True}, smoke={'generated_text': 'text'}, ppl=quality,
                    decode={'median_round_mean_seconds': 2.0, 'model_memory': {'fixture': True}}, score=score)
                with patch.object(engine, 'evaluate_submission', return_value=measured):
                    result = public.evaluate(Path('submission'))
                    self.assertEqual(result['public_reward'], .25)
                    for field in common:
                        self.assertIn(field, result)
                    is_qwen8 = task.name == 'compress-qwen3.8-27b-8g'
                    is_reference = 'qwen3.8' not in task.name
                    for field in penalty:
                        self.assertEqual(field in result, not is_qwen8)
                    self.assertEqual('public_reference_ppl' in result, not is_qwen8)
                    self.assertEqual('speed_pass' in result, is_reference)
                    self.assertEqual('model_memory' in result, not is_reference)
                with patch.object(engine, 'measure_ppl', return_value={'ppl': 2.0}):
                    result = engine.measure_quality(Path('submission'), Path('tokenizer'), target / 'evaluator-assets')
                    expected = task.name.endswith(('16g', '20g', '24g', '32g')) or task.name == 'compress-qwen3.8-27b-12g'
                    self.assertEqual('quality_reference_ppl' in result, expected)
                    self.assertEqual('reference_validation' in result, expected)

    def test_open_payload_formats_keep_package_boundaries(self):
        submission = self.work / 'submission'
        model = submission / 'model'
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        inference = submission / 'inference.py'
        inference.write_text('# package fixture')
        for task in TASKS:
            for role, stage_files in [('public', stage_checker), ('hidden', stage_verifier)]:
                target = self.work / (task.name + '-' + role)
                stage_files(task, target)
                with self.subTest(task=task.name, role=role), entries(target) as (engine, _):
                    for name in ('weights.safetensors', 'weights.gguf', 'weights.bin', 'weights.pt',
                                 'packed.custom', 'packed', 'nested/weights.custom'):
                        payload = model / name
                        payload.parent.mkdir(exist_ok=True)
                        payload.write_bytes(b'opaque payload')
                        report = engine.inspect_submission(submission, engine.DEFAULT_LIMIT_GIB)
                        self.assertTrue(report['valid'])
                        self.assertEqual(report['model_memory_status'], 'not_checked_by_package_inspection')
                        exact = report['total_bytes'] / engine.GIB
                        engine.inspect_submission(submission, exact)
                        with self.assertRaisesRegex(engine.SubmissionError, 'too large'):
                            engine.inspect_submission(submission, (report['total_bytes'] - 1) / engine.GIB)
                        payload.unlink()
                    with self.assertRaisesRegex(engine.SubmissionError, 'no model payload'):
                        engine.inspect_submission(submission, engine.DEFAULT_LIMIT_GIB)
                    inference.unlink()
                    with self.assertRaisesRegex(engine.SubmissionError, 'inference.py'):
                        engine.inspect_submission(submission, engine.DEFAULT_LIMIT_GIB)
                    inference.write_text('# package fixture')
                    (model / 'link.custom').symlink_to(inference)
                    with self.assertRaisesRegex(engine.SubmissionError, 'symbolic links'):
                        engine.inspect_submission(submission, engine.DEFAULT_LIMIT_GIB)
                    (model / 'link.custom').unlink()

    def test_invalid_entry_configuration_is_rejected(self):
        target = self.work / 'checker'
        stage_checker(TASKS[0], target)
        path = target / 'runtime_config.json'
        cfg = json.loads(path.read_text())
        for update in ({'reference_preflight': 'unknown'},
                       {'hidden_reference_fields': 'true'}):
            with self.subTest(update=update):
                path.write_text(json.dumps({**cfg, **update}))
                with self.assertRaises(ValueError), entries(target):
                    pass


if __name__ == '__main__':
    unittest.main()
