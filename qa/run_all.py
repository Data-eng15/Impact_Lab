"""One-command QA run: execute every layer, then build the Excel report.

    python qa/run_all.py              # everything
    python qa/run_all.py --fast       # skip Selenium (no browser needed)
    python qa/run_all.py --layer unit # one layer only

Exit code is non-zero if any P0 case failed, so this can gate CI.
"""
import argparse
import subprocess
import sys
from pathlib import Path

QA = Path(__file__).parent
PY = sys.executable

LAYERS = {
    "unit": "test_unit.py",
    "integration": "test_integration.py",
    "functional": "test_functional_ui.py",
    "nonfunctional": "test_nonfunctional.py",
    "security": "test_security.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the Selenium layer")
    ap.add_argument("--layer", choices=sorted(LAYERS), help="run a single layer")
    args = ap.parse_args()

    if args.layer:
        files = [LAYERS[args.layer]]
    else:
        files = [f for k, f in LAYERS.items() if not (args.fast and k == "functional")]

    targets = [str(QA / f) for f in files]
    print(f"[qa] running {len(targets)} layer(s): {', '.join(files)}\n")

    # Not --exitfirst: we want a complete picture in the report, not the first failure.
    proc = subprocess.run([PY, "-m", "pytest", *targets, "-q", "--no-header"], cwd=QA.parent)

    print()
    report = subprocess.run([PY, str(QA / "build_report.py")], cwd=QA.parent)

    if report.returncode != 0:
        print("\n[qa] P0 FAILURE - do not demo until resolved. See the Defects sheet.")
        return 2
    if proc.returncode != 0:
        print("\n[qa] Non-blocking failures present. See the Defects sheet.")
        return 1
    print("\n[qa] All green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
