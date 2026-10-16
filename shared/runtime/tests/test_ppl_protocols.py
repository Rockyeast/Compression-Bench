"""Exercise multiple PPL protocols in one imported evaluator without a GPU."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from test_task_tools import ROOT, load, stage_checker


class PPLProtocolsTest(unittest.TestCase):
    def test_one_module_switches_formal_debug_and_chat_protocols(self):
        # Import directly from source: no adjacent task configuration is required.
        ppl = load(ROOT / 'shared/evaluator/ppl_benchmark.py')
        with tempfile.TemporaryDirectory() as folder:
            targets = {}
            cases = [
                ('compress-qwen3.6-35b-a3b-12g', '', 18),
                ('compress-qwen3.6-35b-a3b-12g', 'debug-9b', 1),
                ('compress-gemma4-26b-a4b-10g', '', 19),
                ('compress-gemma4-31b-12g', '', 19),
                ('compress-qwen3.8-27b-12g', '', 1),
                ('compress-qwen3.6-35b-a3b-12g', '', 18),
            ]
            for name, debug, start in cases:
                if name not in targets:
                    target = Path(folder) / name
                    stage_checker(ROOT / 'tasks' / name, target)
                    targets[name] = target / 'evaluator-assets'
                assets = targets[name] / debug
                split = debug or 'public'
                consumed = []

                class Runtime:
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def memory_report(self): return None
                    def token_logprobs(self, batch):
                        consumed.extend(batch)
                        return [[None] + [-99.] * (start - 1) + [-.25] * (2048 - start)
                                for _ in batch]

                with self.subTest(task=name, split=split):
                    result = ppl.measure(Path(folder), assets, assets / 'tokenizer', Runtime,
                                         expected_split=split)
                    cfg = json.loads((assets / 'ppl_config.json').read_text())
                    self.assertEqual(consumed, json.loads((assets / 'ppl_inputs.json').read_text()))
                    self.assertEqual(result['ppl_protocol'], cfg['protocol'])
                    self.assertEqual(result['predicted_tokens'], 256 * (2048 - start))
                    self.assertAlmostEqual(result['ppl'], math.exp(.25))
                    ppl.validate_result(result, assets, expected_split=split)
                    for change in ({'ppl_protocol': 'wrong'}, {'ppl_split': 'hidden'},
                                   {'predicted_tokens': 1}):
                        with self.assertRaises(ValueError):
                            ppl.validate_result({**result, **change}, assets, expected_split=split)
            assets = targets['compress-qwen3.6-35b-a3b-12g']
            config_path = assets / 'ppl_config.json'
            cfg = json.loads(config_path.read_text())
            for change in ({'protocol': 'unknown'}, {'scored_tokens': 1},
                           {'context_length': 2049}, {'split': 'hidden'},
                           {'prefix_tokens': [0] * 18}):
                with self.subTest(change=change):
                    config_path.write_text(json.dumps({**cfg, **change}))
                    with self.assertRaises(ValueError):
                        ppl.load_inputs(assets, expected_split='public')


if __name__ == '__main__':
    unittest.main()
