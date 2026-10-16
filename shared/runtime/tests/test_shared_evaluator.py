"""CPU regressions for task-preserving inference and storage sharing."""
import contextlib
import json
import math
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_task_tools import ROOT, TASKS, load, stage_assets, stage_checker

CHAT_TASKS = {
    'compress-gemma4-26b-a4b-16g',
    'compress-gemma4-31b-20g',
    'compress-qwen3.6-35b-a3b-20g',
}


@contextlib.contextmanager
def gguf_placeholder_type():
    """Stand in for the optional plugin; use real PyTorch storage underneath."""
    import torch
    name = 'vllm_gguf_plugin.quantization.params'
    cls = type('GGUFUninitializedWeightTypeParameter',
               (torch.nn.parameter.UninitializedParameter,), {'__module__': name})
    module = types.ModuleType(name)
    module.GGUFUninitializedWeightTypeParameter = cls
    with patch.dict(sys.modules, {name: module}):
        yield cls


class SharedEvaluatorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.memory = load(ROOT / 'shared/evaluator/model_memory.py')

    def test_shared_links_match_docker_sources(self):
        contexts = {'common-assets': ROOT / 'shared/evaluator-assets/common',
                    'public-assets': ROOT / 'shared/evaluator-assets/public',
                    'hidden-assets': ROOT / 'shared/evaluator-assets/hidden'}
        for task in TASKS:
            self.assertFalse((task / 'evaluator/model_memory_policy.json').is_symlink())
            for role, destination in [('environment/public-checker', Path('/checker')), ('tests', Path('/tests'))]:
                with self.subTest(task=task.name, role=role):
                    projected = {}
                    docker = (task / role / 'Dockerfile').read_text()
                    for line in docker.splitlines():
                        if not line.startswith('COPY --from='):
                            continue
                        parts = shlex.split(line)
                        context = parts[1].removeprefix('--from=')
                        if context not in contexts:
                            continue
                        self.assertNotEqual(context, 'hidden-assets' if role.startswith('environment') else 'public-assets')
                        source, output = contexts[context] / parts[2], Path(parts[3])
                        sources = list(source.rglob('*')) if source.is_dir() else [source]
                        for path in sources:
                            if not path.is_file():
                                continue
                            dest = (output / path.relative_to(source) if source.is_dir() else
                                    output / path.name if parts[3].endswith('/') else output)
                            projected[dest] = path.resolve()
                    candidates = [task / 'evaluator/runtime_config.json', *(task / role / 'evaluator-assets').rglob('*')]
                    for link in candidates:
                        if not link.is_symlink():
                            continue
                        self.assertTrue(link.exists(), str(link))
                        self.assertFalse(link.readlink().is_absolute(), str(link))
                        self.assertTrue(link.resolve().is_relative_to(ROOT / 'shared/evaluator-assets'))
                        output = (destination / 'runtime_config.json' if link.name == 'runtime_config.json' else
                                  destination / 'evaluator-assets' / link.relative_to(task / role / 'evaluator-assets'))
                        sources = list(link.rglob('*')) if link.is_dir() else [link]
                        for path in sources:
                            if path.is_file():
                                dest = output / path.relative_to(link) if link.is_dir() else output
                                self.assertEqual(projected[dest], path.resolve())

    def test_smoke_prompt_tokens_preserved_for_every_task(self):
        from unittest.mock import Mock
        for task in TASKS:
            with self.subTest(task=task.name):
                target = self.work / task.name
                stage_checker(task, target)
                runtime = load(target / 'inference_runtime.py')
                tokenizer = Mock()
                tokenizer.return_value.input_ids = [11, 12]
                tokenizer.apply_chat_template.return_value = [21, 22, 23]
                tokenizer.decode.return_value = 'answer'
                tokenizer.__len__ = lambda _: 100
                transformers = types.ModuleType('transformers')
                transformers.AutoTokenizer = Mock()
                transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
                engine = Mock()
                engine.generate.return_value = [[42]]
                factory = Mock()
                factory.return_value.__enter__ = Mock(return_value=engine)
                factory.return_value.__exit__ = Mock(return_value=False)
                with patch.dict(sys.modules, {'transformers': transformers}), patch.object(runtime, 'SubmittedRuntime', factory):
                    self.assertEqual(runtime.generate_smoke('submission', 'tokenizer', 'hello', 4, 60), 'answer')
                if task.name in CHAT_TASKS:
                    tokenizer.apply_chat_template.assert_called_once_with(
                        [{'role': 'user', 'content': 'hello'}], tokenize=True,
                        return_dict=False, add_generation_prompt=True, enable_thinking=False)
                    tokenizer.assert_not_called()
                    ids = [21, 22, 23]
                else:
                    tokenizer.assert_called_once_with('hello', add_special_tokens=False)
                    tokenizer.apply_chat_template.assert_not_called()
                    ids = [11, 12]
                engine.generate.assert_called_once_with([ids], max_new_tokens=4, vocab_size=100)

    def test_gemma_ppl_keeps_prefix_and_scores_only_continuation(self):
        for task in TASKS:
            if not task.name.startswith('compress-gemma'):
                continue
            target = self.work / task.name
            stage_checker(task, target)
            ppl = load(target / 'ppl_benchmark.py')
            for split, folder in [('public', 'environment/public-checker'), ('hidden', 'tests')]:
                with self.subTest(task=task.name, split=split):
                    assets = self.work / (task.name + '-' + split + '-assets')
                    stage_assets(task, folder, assets)
                    original = json.loads((assets / 'ppl_inputs.json').read_text())
                    cfg, rows, _ = ppl.load_inputs(assets, expected_split=split)
                    self.assertEqual(rows, original)
                    self.assertEqual(len(cfg['prefix_tokens']), 19)
                    self.assertTrue(all(row[:19] == cfg['prefix_tokens'] for row in rows))
                    consumed = []
                    class Runtime:
                        def __init__(self, *args, **kwargs): pass
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                        def token_logprobs(self, batch):
                            consumed.extend(batch)
                            # A prefix accidentally included in the score changes PPL sharply.
                            return [[-99.0] * 19 + [-0.25] * 2029 for _ in batch]
                        def memory_report(self): return None
                    result = ppl.measure(self.work, assets, assets / 'tokenizer', Runtime, expected_split=split)
                    self.assertEqual(consumed, rows)
                    self.assertEqual(result['predicted_tokens'], 256 * 2029)
                    self.assertAlmostEqual(result['ppl'], math.exp(.25))

    def test_shared_inputs_match_every_task_and_keep_splits_separate(self):
        for task in TASKS:
            checker = self.work / task.name
            stage_checker(task, checker)
            ppl = load(checker / 'ppl_benchmark.py')
            for role, split in [('environment/public-checker', 'public'), ('tests', 'hidden')]:
                with self.subTest(task=task.name, split=split):
                    target = self.work / (task.name + '-' + split)
                    target.mkdir()
                    assets = target / 'evaluator-assets'
                    stage_assets(task, role, assets)
                    cfg, rows, _ = ppl.load_inputs(assets, expected_split=split)
                    self.assertEqual(len(rows), 256)
                    self.assertEqual(self.memory.read_policy(target),
                                     json.loads((task / 'evaluator/model_memory_policy.json').read_text()))
                    for path in assets.rglob('*.json'):
                        resolved = path.resolve()
                        if resolved.is_relative_to(ROOT / 'shared/evaluator-assets'):
                            self.assertTrue(resolved.is_relative_to(ROOT / 'shared/evaluator-assets' / split))
                    profiles = [assets]
                    if split == 'public':
                        self.assertTrue((assets / 'math500_probe.json').is_file())
                        declared = json.loads((assets / 'evaluation_profiles.json').read_text())
                        self.assertEqual('debug-9b' in declared, (assets / 'debug-9b').is_dir())
                        if 'debug-9b' in declared:
                            debug = assets / 'debug-9b'
                            ppl.load_inputs(debug, expected_split='debug-9b')
                            profiles.append(debug)
                    else:
                        self.assertFalse((assets / 'math500_probe.json').exists())
                        self.assertFalse((assets / 'debug-9b').exists())
                    for profile in profiles:
                        config = json.loads((profile / 'decode_config.json').read_text())
                        prompts = json.loads((profile / 'decode_inputs.json').read_text())
                        self.assertEqual(len(prompts['measured']), config['requests_per_round'])
                        for ids in [prompts['warmup'], *prompts['measured']]:
                            self.assertEqual(len(ids), config['input_tokens'])
                            self.assertTrue(all(type(token) is int and token >= 0 for token in ids))

    def test_storage_aliases_copies_caches_and_both_report_formats(self):
        import torch
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(4))
        model.alias = model.weight[:1]
        model.repacked = model.weight.detach().clone()
        model.aux = torch.zeros(3)
        cache = torch.zeros(10)
        model.cache = cache
        for compatible in (False, True):
            result = self.memory.inventory_model(model, excluded=[cache], device_type='cpu',
                                                 allow_empty_gguf_placeholders=compatible)
            self.assertEqual(result['bytes'], 44)
            self.assertEqual(result['storage_count'], 3)
            self.assertEqual('empty_gguf_type_placeholders' in result, compatible)
        with self.assertRaisesRegex(ValueError, 'aliases a model parameter'):
            self.memory.inventory_model(model, excluded=[model.weight], device_type='cpu')

    def test_only_exact_empty_gguf_metadata_is_exempt(self):
        import torch
        with gguf_placeholder_type() as cls:
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.zeros(4))
            model.weight_type = cls()
            model.weight_type.aux = torch.zeros(3)
            result = self.memory.inventory_model(model, device_type='cpu', allow_empty_gguf_placeholders=True)
            self.assertEqual(result['bytes'], 28)
            self.assertEqual(len(result['empty_gguf_type_placeholders']), 1)
            with self.assertRaises(ValueError):
                self.memory.inventory_model(model, device_type='cpu')
            # A zero-element view can still hold a nonempty allocation.
            model.weight_type.data = torch.zeros(5)[:0]
            with self.assertRaisesRegex(ValueError, 'retains storage'):
                self.memory.inventory_model(model, device_type='cpu', allow_empty_gguf_placeholders=True)
            fake = type(cls.__name__, (torch.nn.parameter.UninitializedParameter,), {'__module__': cls.__module__})
            model.weight_type = fake()
            with self.assertRaisesRegex(ValueError, 'untrusted GGUF'):
                self.memory.inventory_model(model, device_type='cpu', allow_empty_gguf_placeholders=True)
            model.weight_type = torch.nn.parameter.UninitializedParameter()
            with self.assertRaises(ValueError):
                self.memory.inventory_model(model, device_type='cpu', allow_empty_gguf_placeholders=True)

    def test_worker_uses_task_owned_gguf_setting(self):
        import torch
        from unittest.mock import Mock
        for task in TASKS:
            with self.subTest(task=task.name):
                target = self.work / task.name
                stage_checker(task, target)
                memory = load(target / 'model_memory.py')
                model, caches = object(), object()
                worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model, kv_caches=caches), rank=0)
                inventory = Mock(return_value={'bytes': 16})
                with patch.object(torch.cuda, 'synchronize'), patch.object(memory, 'inventory_model', inventory):
                    self.assertEqual(memory.MemoryWorkerExtension.qb_resident_model_inventory(worker)['worker_rank'], 0)
                inventory.assert_called_once_with(model, excluded=caches,
                    allow_empty_gguf_placeholders=not task.name.startswith('compress-qwen3.8'))

    def test_memory_gate_still_requires_load_and_post_inference(self):
        policy = {'protocol': self.memory.PROTOCOL, 'limit_gib': 1}
        def snapshot(amount, phase):
            return {'protocol': self.memory.PROTOCOL, 'phase': phase,
                    'workers': [{'bytes': amount, 'storages': [{'bytes': amount}]}]}
        report = {'protocol': self.memory.PROTOCOL, 'snapshots': [snapshot(16, 'load')]}
        with self.assertRaisesRegex(ValueError, 'load and post-inference'):
            self.memory.validate_report(report, policy)
        report['snapshots'].append(snapshot(2**30 + 1, 'token_logprobs'))
        with self.assertRaisesRegex(ValueError, 'exceeds limit'):
            self.memory.validate_report(report, policy)
        report['snapshots'][-1] = snapshot(32, 'token_logprobs')
        self.memory.validate_report(report, policy)

    def test_isolated_worker_imports_and_inference_protocol(self):
        target = self.work / 'checker'
        stage_checker(TASKS[0], target)
        candidate = self.work / 'submission'
        candidate.mkdir()
        (candidate / 'inference.py').write_text('''
INFERENCE_API_VERSION = 1
class Engine:
    def generate(self, token_ids, **kwargs): return [[7]]
    def token_logprobs(self, token_ids): return [[None, -0.25]]
def load(*args, **kwargs): return Engine()
''')
        requests = [
            {'operation': 'load', 'options': {}},
            {'operation': 'generate', 'token_ids': [[1, 2]], 'max_new_tokens': 1, 'ignore_eos': True},
            {'operation': 'token_logprobs', 'token_ids': [[1, 2]]},
        ]
        result = subprocess.run([sys.executable, '-I', '-B', str(target / 'inference_runtime.py'), '--worker', str(candidate)],
                                input=''.join(json.dumps(r) + '\n' for r in requests),
                                text=True, capture_output=True, cwd=self.work, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(replies, [
            {'ok': True, 'result': 'submitted-inference-audited-v1'},
            {'ok': True, 'result': [[7]]},
            {'ok': True, 'result': [[None, -.25]]},
        ])


if __name__ == '__main__':
    unittest.main()
