"""Build the JupyterLite site used by the presentation.

Requires, in the interpreter that runs this script:

    pip install jupyterlite-core jupyterlite-pyodide-kernel jupyter_server build ipywidgets anywidget

Wheels come from the local braketlab and bubblebox checkouts. Notebooks that
need PySCF or a webcam are left out. The lite kernel installs the wheels with
deps=False so Numba and Evince are not pulled into Pyodide.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_NOTEBOOKS = ROOT / "oslo_met_notebooks"
LITE_SRC = ROOT / "lite_src"
NOTEBOOK_OUT = LITE_SRC / "notebooks"
WHEEL_OUT = LITE_SRC / "wheels"
LITE_OUT = ROOT / "lite"
BRAKETLAB = pathlib.Path.home() / "projects" / "braketlab"
BUBBLEBOX = pathlib.Path.home() / "projects" / "bubblebox"

EXCLUDED = {
    "oslo_met_pyscf.ipynb",
    "oslo_met_h2_wavefunction.ipynb",
    "entangled_visions_demo.ipynb",
    "oslo_met_waves.ipynb",
}

def wheel_requirement(wheel):
    match = re.fullmatch(r"(.+)-([^-]+)-py3-none-any\.whl", wheel.name)
    if match is None:
        raise SystemExit(f"unexpected wheel name: {wheel.name}")
    return f"{match.group(1)}=={match.group(2)}"


def bootstrap_source(wheels):
    # piplite, not micropip: only piplite reads lite/pypi/all.json.
    # The kernel resolves those index URLs before the worker starts.
    specs = ", ".join(json.dumps(wheel_requirement(wheel)) for wheel in wheels)
    return f"""import piplite
await piplite.install(["numpy", "scipy", "sympy", "matplotlib", "anywidget==0.11.0"])
await piplite.install([{specs}], deps=False)
"""


def bootstrap_cell(wheels):
    source = bootstrap_source(wheels).strip() + "\n"
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def prepare_notebooks(wheels):
    if NOTEBOOK_OUT.exists():
        shutil.rmtree(NOTEBOOK_OUT)
    NOTEBOOK_OUT.mkdir(parents=True)
    kept = []
    for path in sorted(SRC_NOTEBOOKS.glob("*.ipynb")):
        if path.name in EXCLUDED:
            continue
        notebook = json.loads(path.read_text())
        cells = []
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if "pip install" in source:
                continue
            if cell.get("cell_type") == "code":
                lines = source.splitlines(keepends=True)
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("import evince")
                ]
                cell["source"] = lines
                cell["outputs"] = []
                cell["execution_count"] = None
            cells.append(cell)
        notebook["cells"] = [bootstrap_cell(wheels), *cells]
        (NOTEBOOK_OUT / path.name).write_text(json.dumps(notebook))
        kept.append(path.name)
    return kept


def build_wheels():
    if WHEEL_OUT.exists():
        shutil.rmtree(WHEEL_OUT)
    WHEEL_OUT.mkdir(parents=True)
    for project in (BRAKETLAB, BUBBLEBOX):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", str(project), "-w", str(WHEEL_OUT)]
        )
    return sorted(WHEEL_OUT.glob("*.whl"))


def write_config(wheels):
    config = {
        "LiteBuildConfig": {
            "contents": [str(NOTEBOOK_OUT)],
            "output_dir": str(LITE_OUT),
        },
        "PipliteAddon": {
            "piplite_urls": [str(wheel) for wheel in wheels],
        },
    }
    (LITE_SRC / "jupyter_lite_config.json").write_text(json.dumps(config, indent=2))


def build_lite():
    import jupyterlite_core

    labextensions = pathlib.Path(sys.prefix) / "share" / "jupyter" / "labextensions"
    required = {
        "ipywidgets": labextensions / "@jupyter-widgets" / "jupyterlab-manager",
        "anywidget": labextensions / "anywidget",
    }
    missing = [name for name, path in required.items() if not path.is_dir()]
    if missing:
        raise SystemExit(
            "JupyterLite would ship without a frontend for: "
            + ", ".join(missing)
            + ". Install them in this interpreter and rebuild: pip install "
            + " ".join(missing)
        )

    print(
        "BUILDER",
        sys.executable,
        jupyterlite_core.__version__,
        jupyterlite_core.__file__,
        flush=True,
    )
    env = os.environ.copy()
    env["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    if LITE_OUT.exists():
        shutil.rmtree(LITE_OUT)
    subprocess.check_call(
        [sys.executable, "-m", "jupyter", "lite", "build", "--force"],
        cwd=LITE_SRC,
        env=env,
    )
    (LITE_OUT / ".nojekyll").write_text("")
    (ROOT / ".nojekyll").write_text("")


def main():
    LITE_SRC.mkdir(parents=True, exist_ok=True)
    wheels = build_wheels()
    kept = prepare_notebooks(wheels)
    write_config(wheels)
    build_lite()
    print("notebooks:", ", ".join(kept))
    print("wheels:", ", ".join(wheel.name for wheel in wheels))
    print("output:", LITE_OUT)


if __name__ == "__main__":
    main()
