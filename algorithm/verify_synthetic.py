"""Run software-contract tests only; no dataset or trained checkpoint is used."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import unittest

BASE = Path(__file__).resolve().parent
PACKAGES = (
    "spectra_a_prototype", "spectra_rewrite_training",
    "spectra_experiment", "spectra_formal_experiment",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "all"), default="quick")
    args = parser.parse_args()
    sys.path.insert(0, str(BASE))
    # Historical tests import sibling modules by their bare names. Supply the
    # search paths without modifying the archived source or its code hashes.
    for package in PACKAGES:
        sys.path.append(str(BASE / "tools" / package))
    if args.suite == "quick":
        modules = [
            "tools.spectra_experiment.test_evaluation",
            "tools.spectra_experiment.test_training",
            "tools.spectra_rewrite_training.test_adapter",
            "tools.spectra_a_prototype.test_numerics",
            "tools.spectra_formal_experiment.test_formal",
        ]
    else:
        modules = [f"tools.{package}.{path.stem}" for package in PACKAGES
                   for path in sorted((BASE / "tools" / package).glob("test_*.py"))]
    suite = unittest.TestSuite()
    for name in modules:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({"scope": "synthetic software contracts only; not paper accuracy or board measurements",
                      "suite": args.suite, "tests_run": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors),
                      "skipped": len(result.skipped), "wall_seconds": round(time.perf_counter() - start, 3)}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
