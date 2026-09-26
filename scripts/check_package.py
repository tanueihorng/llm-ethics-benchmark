"""Build and install a wheel outside the checkout; preserve all staging files."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parent = ROOT / "tmp" / "package-checks"
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="wheel-", dir=parent))
    source = stage / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "ethical_benchmark", source / "ethical_benchmark",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(stage), str(source)], check=True)
    wheel, = stage.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        expected = {str(p.relative_to(source)) for p in (source / "ethical_benchmark").rglob("*.py")}
        missing = expected - set(archive.namelist())
        if missing:
            raise RuntimeError(f"Wheel omits source modules: {sorted(missing)}")
    environment = stage / "venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                    "--no-deps", str(wheel)], check=True)
    probe = """
import importlib.metadata as m
import ethical_benchmark as package
import ethical_benchmark.constants
assert package.__version__ == m.version('fyp-quant')
print(package.__file__)
print('Installed version:', package.__version__)
"""
    subprocess.run([str(python), "-I", "-c", probe], cwd=stage, check=True)
    print(json.dumps({"wheel": str(wheel), "python_modules_verified": len(expected),
                      "note": "Dependency-free install tests packaging; environment resolution is checked separately."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
