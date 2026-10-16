"""Check staged installation bookkeeping without downloading or compiling tools."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("tool_install", Path(__file__).with_name("install.py"))
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class StagedInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logs = Path(self.temp.name)
        self.calls = []
        self.config = json.loads(Path(__file__).with_name("tools.json").read_text())
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(installer, "LOGS", self.logs))
        self.stack.enter_context(patch.object(installer.metadata, "version", return_value="test"))
        self.stack.enter_context(patch.object(installer, "share_nccl"))
        self.stack.enter_context(patch.object(installer, "install_tool", side_effect=self.install))
        self.native = self.stack.enter_context(patch.object(installer.subprocess, "run"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def install(self, name, settings, base):
        self.calls.append(name)
        return name, {"installed": True, "smoke": "passed"}

    def invoke(self, *args):
        with patch.object(installer.sys, "argv", ["install.py", *args]):
            installer.main()

    def test_stages_preserve_all_results_and_allow_native_build(self):
        python_tools = {n for n, c in self.config.items() if not c.get("native_source")}
        native_tools = set(self.config) - python_tools
        self.invoke("--skip-native", "--tool-group", "python")
        self.assertEqual(set(self.calls), python_tools)
        self.assertEqual(set(json.loads((self.logs / "status.json").read_text())), python_tools)
        self.calls.clear()
        self.invoke("--skip-native", "--tool-group", "native")
        self.assertEqual(set(self.calls), native_tools)
        self.assertEqual(set(json.loads((self.logs / "status.json").read_text())), set(self.config))
        self.invoke("--native-only")
        self.native.assert_called_once()

    def test_missing_tool_blocks_shared_native_stage(self):
        (self.logs / "status.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "must pass preparation"):
            self.invoke("--native-only")
        self.native.assert_not_called()

    def test_failed_smoke_blocks_even_installed_tool(self):
        status = {n: {"installed": True, "smoke": "passed"} for n in self.config}
        status["awq"]["smoke"] = "failed"
        (self.logs / "status.json").write_text(json.dumps(status))
        with self.assertRaisesRegex(RuntimeError, "awq"):
            self.invoke("--native-only")
        self.native.assert_not_called()


class SourceBuildTests(unittest.TestCase):
    def test_build_copy_is_writable_and_does_not_change_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            code = source / "setup.py"
            code.write_text("original")
            code.chmod(0o444)
            (source / "build").mkdir()
            (source / "build" / "stale.o").write_text("old build output")
            (source / "sample.txt").write_text("runtime data")
            with tempfile.TemporaryDirectory(dir=root) as scratch:
                copy = Path(installer.prepare_source(str(source), Path(scratch)))
                self.assertTrue(copy.stat().st_mode & 0o200)
                self.assertTrue((copy / "setup.py").stat().st_mode & 0o200)
                (copy / "setup.py").write_text("patched by native build")
                self.assertEqual(code.read_text(), "original")
                self.assertEqual(code.stat().st_mode & 0o222, 0)
                self.assertFalse((copy / "build").exists())
                self.assertEqual((copy / "sample.txt").read_text(), "runtime data")
            self.assertFalse(copy.exists())
            self.assertTrue((source / "build" / "stale.o").exists())

    def test_git_revision_is_preserved(self):
        source = "git+https://example.invalid/repo.git@0123456789abcdef"
        with tempfile.TemporaryDirectory() as scratch:
            self.assertEqual(installer.prepare_source(source, Path(scratch)), source)
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_missing_local_source_fails(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(FileNotFoundError):
                installer.prepare_source(str(Path(scratch) / "missing"), Path(scratch))

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("pip", "setuptools", "wheel")),
                         "offline wheel smoke requires local pip/setuptools/wheel")
    def test_installed_package_survives_disposable_source_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            (source / "setup.py").write_text(
                'from setuptools import setup\n'
                'setup(name="quantbench-source-smoke", version="0.0.1", '
                'py_modules=["quantbench_source_smoke"])\n')
            (source / "quantbench_source_smoke.py").write_text("VALUE = 42\n")
            for file in source.iterdir():
                file.chmod(0o444)
            wheels = root / "wheels"
            installed = root / "installed"
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CONFIG_FILE": os.devnull}
            def run(*args):
                result = subprocess.run([sys.executable, *args], env=env, cwd=root,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with tempfile.TemporaryDirectory(dir=root) as scratch:
                copy = installer.prepare_source(str(source), Path(scratch))
                run("-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                    "--no-cache-dir", "--wheel-dir", str(wheels), copy)
            self.assertFalse(Path(copy).exists())
            run("-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", "--no-compile",
                "--target", str(installed), str(next(wheels.glob("*.whl"))))
            run("-I", "-B", "-c", f"import sys;sys.path.insert(0,{str(installed)!r});"
                "import quantbench_source_smoke;assert quantbench_source_smoke.VALUE == 42")
            self.assertFalse(list(installed.rglob("*.pyc")))
            self.assertEqual(sorted(p.name for p in source.iterdir()),
                             ["quantbench_source_smoke.py", "setup.py"])


if __name__ == "__main__":
    unittest.main()
