"""CPU regression checks for shared task tools and task-local evaluation paths.

Run: python3 -m unittest discover -s shared/runtime/tests -v
Requires safetensors and numpy for the inventory check; no model/GPU is loaded.
"""
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'shared/runtime/task_tools'
TASKS = sorted(p.parent.parent for p in (ROOT / 'tasks').glob('*/evaluator/runtime_config.json'))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_assets(task, role, target):
    """Assemble the real Docker asset sources without duplicating tokenizer files."""
    source_assets = task / role / 'evaluator-assets'
    shutil.copytree(source_assets, target, ignore=shutil.ignore_patterns('tokenizer'))
    (target / 'tokenizer').symlink_to(source_assets / 'tokenizer', target_is_directory=True)
    policy = task / 'evaluator/model_memory_policy.json'
    (target / policy.name).symlink_to(policy)
    split = 'public' if role == 'environment/public-checker' else 'hidden'
    destination = Path('/checker' if split == 'public' else '/tests') / 'evaluator-assets'
    for line in (task / role / 'Dockerfile').read_text().splitlines():
        if not line.startswith(f'COPY --from={split}-assets '):
            continue
        _, _, source, output = shlex.split(line)
        source = ROOT / 'shared/evaluator-assets' / split / source
        output_path = target / Path(output).relative_to(destination)
        sources = list(source.iterdir()) if source.is_dir() else [source]
        for path in sources:
            dest = output_path / path.name if source.is_dir() or output.endswith('/') else output_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Configs may be changed in test fixtures; input arrays remain read-only links.
            if dest.is_symlink():
                dest.unlink()
            if path.name.endswith('_config.json') or dest.name == 'evaluation_profiles.json':
                shutil.copyfile(path, dest)
            elif not dest.exists():
                dest.symlink_to(path)


def stage_checker(task, target):
    target.mkdir()
    for folder in (TOOLS / 'checker', ROOT / 'shared/evaluator', task / 'evaluator', task / 'environment/public-checker'):
        for source in folder.glob('*.py'):
            shutil.copyfile(source, target / source.name)
    shutil.copyfile(task / 'evaluator/runtime_config.json', target / 'runtime_config.json')
    stage_assets(task, 'environment/public-checker', target / 'evaluator-assets')


def stage_verifier(task, target):
    target.mkdir()
    for folder in (ROOT / 'shared/evaluator', task / 'evaluator', task / 'tests'):
        for source in folder.glob('*.py'):
            shutil.copyfile(source, target / source.name)
    shutil.copyfile(task / 'evaluator/runtime_config.json', target / 'runtime_config.json')
    shutil.copyfile(ROOT / 'shared/evaluator/test.sh', target / 'test.sh')
    stage_assets(task, 'tests', target / 'evaluator-assets')


class SharedToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def test_all_tasks_package_checks_and_capacity_policy(self):
        submission = self.work / 'submission'
        (submission / 'model').mkdir(parents=True)
        (submission / 'inference.py').write_text('# Package inspection fixture\n')
        (submission / 'model/config.json').write_text('{}')
        # Valid tiny safetensors payload, though package inspection only inventories files.
        import numpy as np
        from safetensors.numpy import save_file
        save_file({'weight': np.zeros((2, 2), dtype=np.float32)}, submission / 'model/model.safetensors')
        for task in TASKS:
            with self.subTest(task=task.name):
                checker = self.work / task.name
                stage_checker(task, checker)
                out = self.work / 'result.json'
                command = [sys.executable, str(checker / 'evaluate_public.py'), str(submission), '--package-only', '--output', str(out)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(out.read_text())['valid'])
                # The shared diagnostic still resolves the task's actual question assets.
                result = subprocess.run([sys.executable, '-c', 'import quality_check; assert len(quality_check.load_questions(32)[1]) == 32; assert len(quality_check.load_questions(64)[1]) == 64'], cwd=checker, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                hidden = self.work / (task.name + '-hidden')
                stage_verifier(task, hidden)
                result = subprocess.run([sys.executable, str(hidden / 'evaluate.py'), 'check', str(submission)], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                hidden_report = json.loads(result.stdout)
                self.assertTrue(hidden_report['valid'])
                public_report = json.loads(out.read_text())
                limit = json.loads((task / 'evaluator/model_memory_policy.json').read_text())['limit_gib']
                self.assertEqual(public_report['limit_gib'], limit)
                self.assertEqual(hidden_report['limit_gib'], limit)
                # Change only the staged policy: both roles must pick it up, while debug stays 32 GiB.
                for directory, report in [(checker, public_report), (hidden, hidden_report)]:
                    policy_path = directory / 'evaluator-assets/model_memory_policy.json'
                    policy = json.loads(policy_path.read_text())
                    policy_path.unlink()  # Staging uses symlinks; never write through to source assets.
                    for bytes_allowed in (report['total_bytes'], report['total_bytes'] - 1):
                        policy['limit_gib'] = bytes_allowed / 1024**3
                        policy_path.write_text(json.dumps(policy))
                        code = r'''import json, sys
from pathlib import Path
import evaluate
limit = json.loads(Path('evaluator-assets/model_memory_policy.json').read_text())['limit_gib']
assert evaluate.DEFAULT_LIMIT_GIB == limit
assert evaluate.model_memory.read_policy(Path.cwd())['limit_gib'] == limit
for arguments in [['check', sys.argv[1]], ['run', sys.argv[1]], ['smoke', sys.argv[1]],
                  ['ppl', sys.argv[1], '--tokenizer', 'evaluator-assets/tokenizer'],
                  ['decode', sys.argv[1], '--ppl-result', 'ppl.json']]:
    assert evaluate.build_parser().parse_args(arguments).limit_gib == limit
assert evaluate.build_parser().parse_args(['check', sys.argv[1], '--limit-gib', '3']).limit_gib == 3
if Path('evaluator-assets/evaluation_profiles.json').exists():
    import evaluate_public
    assert evaluate_public.DEFAULT_PROFILE['size_limit_gib'] == limit
    profiles = json.loads(evaluate_public.PROFILE_CONFIG.read_text())
    if 'debug-9b' in profiles:
        debug_limit = evaluate_public.load_profile('debug-9b')['size_limit_gib']
        assert debug_limit == 32
        with evaluate.model_memory.using_profile_limit(debug_limit):
            assert evaluate.model_memory.read_policy(Path.cwd())['limit_gib'] == 32
        assert evaluate.model_memory.read_policy(Path.cwd())['limit_gib'] == limit
    check = lambda: evaluate_public.evaluate(Path(sys.argv[1]), package_only=True)
else:
    profile = dict(tokenizer=Path('evaluator-assets/tokenizer'), size_limit_gib=limit, score=True)
    check = lambda: evaluate.evaluate_submission(Path(sys.argv[1]), profile, package_only=True)
try:
    check()
except evaluate.SubmissionError as error:
    assert 'too large' in str(error)
    assert sys.argv[2] == 'reject'
else:
    assert sys.argv[2] == 'accept'
'''
                        expected = 'accept' if bytes_allowed == report['total_bytes'] else 'reject'
                        result = subprocess.run([sys.executable, '-c', code, str(submission), expected],
                                                cwd=directory, capture_output=True, text=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_invalid_capacity_policy_fails_closed(self):
        checker = self.work / 'checker'
        stage_checker(TASKS[0], checker)
        path = checker / 'evaluator-assets/model_memory_policy.json'
        policy = json.loads(path.read_text())
        path.unlink()
        for value in (None, True, 0, -1, float('nan')):
            with self.subTest(limit=value):
                if value is not None:
                    path.write_text(json.dumps({**policy, 'limit_gib': value}))
                result = subprocess.run([sys.executable, '-c', 'import evaluate_public'],
                                        cwd=checker, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('policy', result.stderr)

    def test_tokenizer_packaging_preserves_task_rules(self):
        for task in TASKS:
            with self.subTest(task=task.name):
                work = self.work / task.name
                work.mkdir()
                shutil.copyfile(ROOT / 'shared/evaluator/submission_packaging.py', work / 'submission_packaging.py')
                shutil.copyfile(task / 'evaluator/runtime_config.json', work / 'runtime_config.json')
                module = load(work / 'submission_packaging.py')
                official = work / 'official'
                official.mkdir()
                expected_files = ['tokenizer.json', 'tokenizer_config.json']
                if 'qwen3.8' not in task.name:
                    expected_files.append('chat_template.jinja')
                self.assertEqual(list(module.OFFICIAL_FILES), expected_files)
                for name in expected_files:
                    (official / name).write_text('official ' + name)
                submission = work / 'submission'
                payloads = ['model/w.safetensors', 'model/w.gguf', 'model/custom.bin', 'model/nested/packed.custom', 'model/no_extension',
                            'model/helper.py', 'model/notes.txt', 'model/config.json',
                            'outside.safetensors', 'outside.gguf', 'inference.py']
                for name in payloads + ['model/tokenizer.json', 'model/chat_templates/old.jinja']:
                    path = submission / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('candidate ' + name)
                original = {str(p.relative_to(submission)): p.read_bytes() for p in submission.rglob('*') if p.is_file()}
                expected_bytes = sum((submission / p).stat().st_size for p in payloads)
                expected_bytes += sum(p.stat().st_size for p in official.iterdir())
                inventory = module.packaged_inventory(submission, official)
                self.assertEqual(inventory['total_bytes'], expected_bytes)
                with module.prepared_submission(submission, official) as prepared:
                    self.assertFalse((prepared / 'model/chat_templates').exists())
                    for name in payloads:
                        relative = Path(name)
                        linked = name.startswith('model/') and relative.suffix not in ('.json', '.py', '.md', '.txt')
                        self.assertEqual((prepared / name).is_symlink(), linked, name)
                        self.assertEqual((prepared / name).read_bytes(), original[name])
                    for name in expected_files:
                        self.assertEqual((prepared / 'model' / name).read_bytes(), (official / name).read_bytes())
                        self.assertEqual((prepared / 'model' / name).stat().st_mode & 0o777, 0o444)
                self.assertFalse(prepared.exists())
                self.assertEqual({str(p.relative_to(submission)): p.read_bytes() for p in submission.rglob('*') if p.is_file()}, original)
                self.assertEqual(module.finalize_submission(submission, official), inventory)
                self.assertEqual(sum(p.stat().st_size for p in submission.rglob('*') if p.is_file()), expected_bytes)
                (submission / 'model/link').symlink_to(official / expected_files[0])
                with self.assertRaisesRegex(ValueError, 'symbolic links'):
                    module.packaged_inventory(submission, official)
                (submission / 'model/link').unlink()
                (official / expected_files[-1]).unlink()
                with self.assertRaisesRegex(ValueError, 'task-owned tokenizer'):
                    module.packaged_inventory(submission, official)

    def test_invalid_tokenizer_packaging_configuration_fails_closed(self):
        source = self.work / 'submission_packaging.py'
        shutil.copyfile(ROOT / 'shared/evaluator/submission_packaging.py', source)
        config = json.loads((TASKS[0] / 'evaluator/runtime_config.json').read_text())
        for update in [{'tokenizer_files': []}, {'tokenizer_files': '../escape'},
                       {'tokenizer_files': ['../escape']}, {'tokenizer_files': ['.']},
                       {'tokenizer_files': ['tokenizer.json', 'tokenizer.json']},
                       {'tokenizer_files': [None]}]:
            with self.subTest(update=update):
                (self.work / 'runtime_config.json').write_text(json.dumps({**config, **update}))
                with self.assertRaises(ValueError):
                    load(source)

    def start_server(self, evaluator):
        sock = self.work / 'checker.sock'
        proc = subprocess.Popen([sys.executable, str(TOOLS / 'checker/server.py'), '--socket', str(sock), '--submission', '/submission', '--evaluator', str(evaluator), '--evaluation-timeout', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def stop():
            proc.terminate()
            proc.wait(timeout=5)
        self.addCleanup(stop)
        until = time.monotonic() + 5
        while not sock.exists() and proc.poll() is None and time.monotonic() < until:
            time.sleep(.01)
        self.assertTrue(sock.exists(), 'Checker failed to start')
        return sock

    def client(self, sock, flags, *, debug=False):
        with patch.dict(os.environ, {'QUANT_BENCH_DEBUG_9B': '1' if debug else '0'}):
            client = load(TOOLS / 'agent/check_submission.py')
            client.SOCKET_PATH = sock
            output = io.StringIO()
            with patch.object(sys, 'argv', ['check_submission.py', *flags]), contextlib.redirect_stdout(output):
                client.main()
            return json.loads(output.getvalue())

    def test_socket_client_server_modes_and_invalid_requests(self):
        evaluator = self.work / 'fake_evaluator.py'
        evaluator.write_text('''import argparse,json
p=argparse.ArgumentParser()
p.add_argument('submission'); p.add_argument('--output'); p.add_argument('--profile')
p.add_argument('--package-only',action='store_true'); p.add_argument('--quality-only',action='store_true'); p.add_argument('--questions',type=int)
a=p.parse_args()
json.dump(dict(valid=True,submission=a.submission,profile=a.profile,package_only=a.package_only,quality_only=a.quality_only,questions=a.questions,public_reward=.5),open(a.output,'w'))
''')
        sock = self.start_server(evaluator)
        for mode in ([], ['--package-only'], ['--quality-only', '--questions', '64']):
            with self.subTest(mode=mode):
                result = self.client(sock, ['/app/submission', *mode])
                self.assertEqual(result['submission'], '/submission')
                self.assertEqual(result['package_only'], '--package-only' in mode)
                self.assertEqual(result['quality_only'], '--quality-only' in mode)
                if '--quality-only' in mode:
                    self.assertEqual(result['questions'], 64)
        result = self.client(sock, ['/app/debug-submission', '--debug-9b'], debug=True)
        self.assertEqual(result['submission'], '/debug-submission')
        self.assertEqual(result['profile'], 'debug-9b')
        self.assertTrue(result['debug_only'])
        self.assertNotIn('public_reward', result)
        for flags in (['/app/submission', '--debug-9b'], ['/app/submission', '--questions', '64'], ['/tmp/wrong-submission']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.client(sock, flags)
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(str(sock))
            connection.sendall(b'{"protocol":1,"action":"delete"}\n')
            response = load(TOOLS / 'agent/check_submission.py').receive_line(connection)
        self.assertFalse(json.loads(response)['valid'])
        self.assertTrue(self.client(sock, ['/app/submission', '--package-only'])['valid'])

    def test_evaluator_timeout_and_nonzero_exit(self):
        server = load(TOOLS / 'checker/server.py')
        evaluator = self.work / 'fail.py'
        evaluator.write_text('import sys; sys.exit(1)')
        self.assertFalse(server.run_evaluator(evaluator, self.work, True, 1)['valid'])
        evaluator.write_text('import time; time.sleep(5)')
        with self.assertRaises(subprocess.TimeoutExpired):
            server.run_evaluator(evaluator, self.work, True, .1)

    def test_response_limits(self):
        class Connection:
            def __init__(self, count): self.count = count
            def recv(self, size):
                amount = min(self.count, size)
                self.count -= amount
                return b'x' * amount
        for limit in (1024 * 1024, 64 * 1024 * 1024):
            with self.subTest(limit=limit), patch.dict(os.environ, {'QUANT_BENCH_CHECKER_MAX_RESPONSE_BYTES': str(limit)}):
                client = load(TOOLS / 'agent/check_submission.py')
                self.assertEqual(len(client.receive_line(Connection(limit))), limit)
                with self.assertRaises(RuntimeError):
                    client.receive_line(Connection(limit + 1))

    def test_submission_guard_blocks_exclusive_verification(self):
        lock = self.work / 'lock'
        lock.touch()
        command = [sys.executable, str(TOOLS / 'agent/submission_guard.py')]
        proc = subprocess.Popen([*command, 'hold', '--lock', str(lock)])
        try:
            until = time.monotonic() + 5
            while time.monotonic() < until:
                if subprocess.run([*command, 'ready', '--lock', str(lock)]).returncode == 0:
                    break
            else:
                self.fail('Guard failed to acquire lock')
            with lock.open('rb') as handle, self.assertRaises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            proc.terminate()
            proc.wait(timeout=5)
        self.assertEqual(subprocess.run([*command, 'ready', '--lock', str(lock)]).returncode, 1)

    def test_cpu_inventory(self):
        import numpy as np
        from safetensors.numpy import save_file
        inventory = load(TOOLS / 'agent/model_inventory.py')
        save_file({'layer.0.weight': np.ones((2, 3), dtype=np.float32)}, self.work / 'model.safetensors')
        report = inventory.scan(self.work)
        self.assertEqual(report['tensor_bytes'], 24)
        self.assertEqual(report['tensor_count'], 1)
        (self.work / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'missing': 'model.safetensors'}}))
        with self.assertRaises(ValueError):
            inventory.scan(self.work)


if __name__ == '__main__':
    unittest.main()
