"""Install isolated compression tools and their native extensions into the Agent image."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import filecmp
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

PROTECTED = ("torch", "transformers", "compressed-tensors", "vllm", "humming-kernels")
LOGS = Path("/opt/compression-install-logs")

def validate_config(config):
    if not isinstance(config, dict) or not config:
        raise ValueError("Expected a nonempty mapping of tool names to installation rules")
    for name, tool in config.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "quantize":
            raise ValueError(f"Invalid or reserved tool environment: {name}")
        if not {"requirements", "overrides", "smoke_test"} <= set(tool):
            raise ValueError(f"{name}: expected requirements, overrides, smoke_test")
        pins = {}
        for requirement in tool["requirements"]:
            if not isinstance(requirement, str) or not re.fullmatch(r"[a-z0-9-]+==[A-Za-z0-9.+!-]+", requirement):
                raise ValueError(f"{name}: dependencies must use exact package==version pins")
            package, version = requirement.split("==")
            if package in pins:
                raise ValueError(f"{name}: duplicate dependency {package}")
            pins[package] = version
        if not pins and not (tool.get("source_install") or tool.get("package_no_deps")):
            raise ValueError(f"{name}: no requirements")
        overrides = tool["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"{name}: overrides must be a mapping")
        for package, version in overrides.items():
            if package not in PROTECTED or pins.get(package) != version:
                raise ValueError(f"{name}: overrides must match explicit protected-package pins")
        if not isinstance(tool["smoke_test"], str) or not tool["smoke_test"].strip():
            raise ValueError(f"{name}: smoke_test is required")
        compile(tool["smoke_test"], f"<{name} smoke test>", "exec")



def share_nccl(name):
    for binary in Path('/opt', name).glob('lib/python*/site-packages/nvidia/nccl/lib/libnccl.so*'):
        if binary.is_symlink():
            continue
        # Version is metadata only; compare content before replacing any duplicate.
        site = binary.parents[3]
        metadata = list(site.glob('nvidia_nccl_cu*-*.dist-info'))
        if len(metadata) != 1:
            continue  # Multiple installed variants cannot be safely attributed to this binary.
        version = metadata[0].name.removeprefix('nvidia_nccl_cu13-').removesuffix('.dist-info')
        if version.startswith('nvidia_nccl_'):  # Keep different CUDA families separate.
            version = version.removeprefix('nvidia_nccl_')
        target = Path('/opt/compression-shared/nccl', version, binary.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not filecmp.cmp(binary, target, shallow=False):
                raise RuntimeError(f'Different NCCL binaries claim the same version: {version}')
            binary.unlink()
        else:
            binary.rename(target)
        binary.symlink_to(target)


def prepare_source(source, scratch):
    """Build bundled projects in disposable copies; keep the mounted library pristine."""
    if not Path(source).is_absolute():
        return source  # Pinned Git URLs are fetched and cleaned up by pip.
    target = Path(tempfile.mkdtemp(prefix="source-", dir=scratch)) / "project"
    shutil.copytree(source, target,
                    ignore=shutil.ignore_patterns("build", "__pycache__", "*.egg-info"))
    # Bundled sources may already be read-only; build backends need a writable copy.
    for path in [target, *target.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    return str(target)


def install_tool(name, config, base):
    started = time.monotonic()
    python = f"/opt/{name}/bin/python"
    expected = {**base, **config["overrides"]}
    pins = dict(item.split("==") for item in config["requirements"])
    for package in PROTECTED:
        if package in pins and pins[package] != expected[package]:
            raise ValueError(f"{name}: changing {package} requires an explicit override")
    status = {"environment": f"/opt/{name}", "installed": False}
    with tempfile.TemporaryDirectory(prefix=f"compression-{name}-") as scratch:
        scratch = Path(scratch)
        constraints = scratch / "constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k, v in expected.items()))
        env = {**os.environ, "MAX_JOBS": "2", "TORCH_CUDA_ARCH_LIST": os.environ["TORCH_CUDA_ARCH_LIST"]}
        with (LOGS / f"{name}.log").open("w") as log:
            def run(args, timeout=3600):
                # Terminate compiler children too if a build exceeds its budget.
                with subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        result = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                    if result:
                        raise subprocess.CalledProcessError(result, args)

            try:
                run([sys.executable, "-m", "venv", "--system-site-packages", f"/opt/{name}"])
                if config["requirements"]:
                    run([python, "-m", "pip", "install", "--no-compile",
                         *(["--no-build-isolation"] if not config.get("build_isolation", True) else []),
                         "-c", str(constraints), *config["requirements"]])
                if config.get("package_no_deps", []):
                    run([python, "-m", "pip", "install", "--no-compile", "--no-deps",
                         *config.get("package_no_deps", [])])
                for source in config.get("source_install", []):
                    run([python, "-m", "pip", "install", "--no-compile", "--no-deps",
                         prepare_source(source, scratch)])
                if config.get("native_source"):
                    source = Path(prepare_source(config["native_source"], scratch))
                    includes = Path("/usr/local/lib/python3.12/dist-packages/nvidia").glob("*/include")
                    env["CPATH"] = ":".join(str(p) for p in includes)
                    env["CPLUS_INCLUDE_PATH"] = env["CPATH"]
                    if config["native_kind"] == "awq":
                        setup = source / "setup.py"
                        setup.write_text(setup.read_text().replace("-std=c++17", "-std=c++20"))
                    elif config["native_kind"] == "bitorch":
                        env.update(BIE_FORCE_CUDA="true", BIE_CUDA_ARCH="sm_" + env["TORCH_CUDA_ARCH_LIST"].replace(".", ""),
                                   BIE_BUILD_ONLY="bitorch_engine/layers/qlinear/nbit/cuda")
                        helper = source / "bitorch_engine/utils/cuda_extension.py"
                        helper.write_text(helper.read_text().replace("if major > 11:", "if False:"))
                    run([python, "-m", "pip", "install", "--no-compile", "--no-deps",
                         "--no-build-isolation", "--verbose", str(source)], timeout=3600)
                if config.get("library_dirs"):
                    site = f"/opt/{name}/lib/python3.12/site-packages"
                    dirs = ":".join(f"{site}/{d}" for d in config["library_dirs"])
                    wrapper = Path(f"/opt/{name}/bin/run-python")
                    wrapper.write_text(f'#!/bin/sh\nexport LD_LIBRARY_PATH={dirs}:${{LD_LIBRARY_PATH:-}}\n'
                                       f'exec {python} "$@"\n')
                    wrapper.chmod(0o755)
                installed_versions = {**expected, **pins, **config.get("installed_versions", {})}
                run([python, "-c", "from importlib.metadata import version;"
                     f"expected={installed_versions!r};actual={{k:version(k) for k in expected}};assert actual==expected, (actual, expected)"])
                status["installed"] = True
                if config.get("defer_smoke"):
                    status["smoke"] = "requires_gpu"
                    status["reason"] = config["defer_smoke"]
                else:
                    run([python, "-c", config["smoke_test"]], timeout=180)
                    status["smoke"] = "passed"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                status["smoke"] = "failed"
                status["error"] = str(error)
                log.flush()
                print((LOGS / f"{name}.log").read_text(errors="replace")[-6000:], flush=True)
            freeze = subprocess.check_output([python, "-m", "pip", "freeze", "--local"], text=True)
            (LOGS / f"{name}-requirements.txt").write_text(freeze)
    status["seconds"] = round(time.monotonic() - started, 1)
    return name, status


NATIVE_EXTENSIONS = r"""
#!/bin/bash
# Build once against this image's Torch/CUDA ABI, then reuse across isolated envs.
set -euo pipefail
wheels=$(mktemp -d)
trap 'rm -rf "$wheels"' EXIT
cp -a /opt/quant-docs/common/upstream/fast-hadamard-transform "$wheels/hadamard-source"
cp -a /opt/quant-docs/quantization/qtip/upstream/qtip-kernels "$wheels/qtip-source"
chmod -R u+w "$wheels/hadamard-source" "$wheels/qtip-source"
# CUDA libraries installed as wheels keep headers outside CUDA_HOME/include.
export CPATH="$(python3 -c 'import site,glob; print(":".join(p for base in site.getsitepackages() for p in glob.glob(base+"/nvidia/*/include")))')${CPATH:+:$CPATH}"
export MAX_JOBS=2
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:?target architecture must be inherited from the runtime image}"
# Current Torch headers require C++20; only patch the temporary build copy.
sed -i 's/-std=c++17/-std=c++20/g' "$wheels/qtip-source/setup.py"
if [[ all != qtip ]]; then
/opt/spinquant/bin/python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  "$wheels/hadamard-source" -w "$wheels"
for tool in spinquant d2quant qtip; do
  /opt/"$tool"/bin/python -m pip install --no-compile --no-deps "$wheels"/*.whl
  /opt/"$tool"/bin/python -c 'import torch; import fast_hadamard_transform'
done
fi
# QTIP's Python code imports its compiled extension unconditionally.
if [[ all != hadamard ]]; then
/opt/qtip/bin/python -m pip install --no-compile --no-deps --no-build-isolation \
  "$wheels/qtip-source"
fi
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate manifest without installing")
    native = parser.add_mutually_exclusive_group()
    native.add_argument("--skip-native", action="store_true", help="Install environments; cache before shared native extensions")
    native.add_argument("--native-only", action="store_true", help="Build shared native extensions in prepared environments")
    parser.add_argument("--tool-group", choices=("all", "python", "native"), default="all",
                        help="Cache Python environments separately from per-tool CUDA builds")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("tools.json").read_text())
    validate_config(config)
    if args.check:
        print(f"Validated {len(config)} tool environments")
        return
    if args.native_only:
        results = json.loads((LOGS / "status.json").read_text())
        missing = [name for name in config if not results.get(name, {}).get("installed")
                   or results[name].get("smoke") not in ("passed", "requires_gpu")]
        if missing:
            raise RuntimeError(f"Tool environments must pass preparation first: {missing}")
        subprocess.run(["bash", "-c", NATIVE_EXTENSIONS], check=True)
        return
    base = {name: metadata.version(name) for name in PROTECTED}
    LOGS.mkdir(exist_ok=True)
    status_file = LOGS / "status.json"
    results = json.loads(status_file.read_text()) if status_file.exists() else {}
    selected = {name: settings for name, settings in config.items()
                if args.tool_group == "all"
                or bool(settings.get("native_source")) == (args.tool_group == "native")}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(install_tool, name, settings, base) for name, settings in selected.items()]
        for future in as_completed(futures):
            name, status = future.result()
            results[name] = status
            print(name, json.dumps(status), flush=True)
            (LOGS / "status.json").write_text(json.dumps(results, indent=2))
    for name, status in results.items():
        if status["installed"]:
            share_nccl(name)
    if {name: metadata.version(name) for name in PROTECTED} != base:
        raise RuntimeError("Base inference package versions changed")
    (LOGS / "base-versions.json").write_text(json.dumps(base, indent=2))
    failures = [name for name, status in results.items() if status["smoke"] == "failed"]
    if failures:
        raise RuntimeError(f"Tool preparation failed; see {LOGS}: {failures}")
    if not args.skip_native:
        subprocess.run(["bash", "-c", NATIVE_EXTENSIONS], check=True)


if __name__ == "__main__":
    main()
