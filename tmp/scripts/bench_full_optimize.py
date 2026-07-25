#!/usr/bin/env python3
"""Full optimization benchmark: compare old vs new GeneticOptimization.optimize_ems().

Runs a complete genetic optimization with a fixed seed on both versions and
compares the final solution output field-by-field.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TESTDATA = ROOT / "tests" / "testdata"
SRC = ROOT / "src"

# ── Load old module ───────────────────────────────────────────────────
old_path = Path("/tmp/old_genetic.py")
spec_old = importlib.util.spec_from_file_location("old_genetic_full", old_path)
old_genetic = importlib.util.module_from_spec(spec_old)  # type: ignore
spec_old.loader.exec_module(old_genetic)  # type: ignore
OldGeneticOptimization = old_genetic.GeneticOptimization

# ── Load new module ───────────────────────────────────────────────────
sys.path.insert(0, str(SRC))
from akkudoktoreos.config.config import ConfigEOS  # noqa: E402
from akkudoktoreos.core.coreabc import get_ems  # noqa: E402
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization  # noqa: E402
from akkudoktoreos.optimization.genetic.geneticparams import (  # noqa: E402
    GeneticOptimizationParameters,
)
from akkudoktoreos.utils.datetimeutil import to_datetime  # noqa: E402

# ── Setup ─────────────────────────────────────────────────────────────
ems_eos = get_ems(init=True)


def run_optimization(
    OptClass: type,  # type: ignore[name-defined]
    input_data: GeneticOptimizationParameters,
    start_hour: int,
    ngen: int,
    fixed_seed: int,
    config: ConfigEOS,
) -> tuple[Any, float]:
    """Run full optimization, return (solution, elapsed_seconds)."""
    # Patch config onto the optimizer class
    opt = OptClass(fixed_seed=fixed_seed)  # type: ignore[call-arg]
    opt.config = config
    opt.ems = ems_eos

    ems_eos.set_start_datetime(to_datetime().set(hour=start_hour))

    t0 = time.perf_counter()
    solution = opt.optimize_ems(
        parameters=input_data,
        start_hour=start_hour,
        ngen=ngen,
    )
    elapsed = time.perf_counter() - t0
    return solution, elapsed


def compare_solutions(old_sol: Any, new_sol: Any, label: str) -> list[str]:
    """Compare two GeneticSolution objects field-by-field."""
    issues: list[str] = []
    old_d = old_sol.model_dump()
    new_d = new_sol.model_dump()

    all_keys = set(old_d.keys()) | set(new_d.keys())
    for key in sorted(all_keys):
        old_v = old_d.get(key)
        new_v = new_d.get(key)

        if key not in old_d:
            issues.append(f"  [MISSING in old]  {key}")
            continue
        if key not in new_d:
            issues.append(f"  [MISSING in new]  {key}")
            continue

        # Skip non-numeric fields that may differ (e.g. start_solution genome)
        if key == "start_solution":
            if len(old_v) != len(new_v):  # type: ignore[arg-type]
                issues.append(f"  {key}: length mismatch old={len(old_v)} new={len(new_v)}")  # type: ignore[arg-type]
            else:
                old_a = np.array(old_v, dtype=float)  # type: ignore[arg-type]
                new_a = np.array(new_v, dtype=float)  # type: ignore[arg-type]
                diffs = np.sum(old_a != new_a)
                if diffs > 0:
                    issues.append(f"  {key}: {diffs} genome positions differ (expected with GA)")
            continue

        if key == "ev_obj":
            # Compare EV object as dict
            if old_v and new_v:
                old_ev = old_v.to_dict() if hasattr(old_v, "to_dict") else old_v
                new_ev = new_v.to_dict() if hasattr(new_v, "to_dict") else new_v
                if old_ev != new_ev:
                    issues.append(f"  {key}: EV objects differ")
            elif old_v != new_v:
                issues.append(f"  {key}: one is None, other is not")
            continue

        if key == "result":
            # Compare simulation result dict
            if isinstance(old_v, dict) and isinstance(new_v, dict):
                result_issues = compare_results(old_v, new_v, key)
                issues.extend(result_issues)
            elif old_v != new_v:
                issues.append(f"  {key}: result objects differ (type: {type(old_v)} vs {type(new_v)})")
            continue

        # Scalar
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            diff = abs(float(old_v) - float(new_v))
            if diff > 1e-9:
                issues.append(f"  {key}: old={old_v!r} new={new_v!r} diff={diff:.2e}")
            continue

        # List/array
        if isinstance(old_v, (list, np.ndarray)) and isinstance(new_v, (list, np.ndarray)):
            oa = np.asarray(old_v, dtype=float)
            na = np.asarray(new_v, dtype=float)
            if oa.shape != na.shape:
                issues.append(f"  {key}: shape mismatch old={oa.shape} new={na.shape}")
                continue
            mask = ~(np.isnan(oa) & np.isnan(na))
            if mask.any():
                max_diff = np.nanmax(np.abs(oa[mask] - na[mask]))
                if max_diff > 1e-9:
                    idx = int(np.nanargmax(np.abs(oa - na)))
                    issues.append(
                        f"  {key}: max_diff={max_diff:.2e} at idx={idx} "
                        f"(old={oa[idx]:.6f}, new={na[idx]:.6f})"
                    )
            continue

        # Fallback
        if str(old_v) != str(new_v):
            issues.append(f"  {key}: old={old_v!r} new={new_v!r}")

    return issues


def compare_results(old_r: dict, new_r: dict, prefix: str) -> list[str]:
    """Compare two simulation result dicts."""
    issues: list[str] = []
    all_keys = set(old_r.keys()) | set(new_r.keys())
    for key in sorted(all_keys):
        old_v = old_r.get(key)
        new_v = new_r.get(key)
        if key not in old_r or key not in new_r:
            issues.append(f"  {prefix}.{key}: presence mismatch")
            continue
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            diff = abs(float(old_v) - float(new_v))
            if diff > 1e-9:
                issues.append(f"  {prefix}.{key}: old={old_v} new={new_v} diff={diff:.2e}")
        elif isinstance(old_v, (list, np.ndarray)) and isinstance(new_v, (list, np.ndarray)):
            oa = np.asarray(old_v, dtype=float)
            na = np.asarray(new_v, dtype=float)
            if oa.shape != na.shape:
                issues.append(f"  {prefix}.{key}: shape {oa.shape} vs {na.shape}")
            else:
                mask = ~(np.isnan(oa) & np.isnan(na))
                if mask.any():
                    max_diff = np.nanmax(np.abs(oa[mask] - na[mask]))
                    if max_diff > 1e-9:
                        issues.append(f"  {prefix}.{key}: max_diff={max_diff:.2e}")
    return issues


def main() -> None:
    start_hour = 10
    fixed_seed = 42
    ngen = 20  # Enough generations for convergence comparison

    # Setup config
    config = ConfigEOS()
    config.merge_settings_from_dict(
        {
            "prediction": {"hours": 48},
            "optimization": {
                "horizon_hours": 48,
                "genetic": {
                    "individuals": 100,
                    "generations": 20,
                    "penalties": {
                        "ev_soc_miss": 10,
                        "ac_charge_break_even": 0,
                    },
                },
            },
            "devices": {
                "max_electric_vehicles": 1,
                "electric_vehicles": [
                    {"charge_rates": [0.0, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]},
                ],
            },
        }
    )

    input_files = [
        TESTDATA / "optimize_input_1.json",
        TESTDATA / "optimize_input_2.json",
    ]

    for inp in input_files:
        print(f"\n{'=' * 70}")
        print(f"File: {inp.name}  (ngen={ngen}, seed={fixed_seed})")
        print(f"{'=' * 70}")

        with inp.open() as f:
            data = GeneticOptimizationParameters(**json.load(f))

        # Run old version
        print("  Running OLD optimization...")
        old_sol, old_time = run_optimization(
            OldGeneticOptimization, data, start_hour, ngen, fixed_seed, config
        )
        print(f"  OLD done in {old_time:.2f}s")

        # Run new version
        print("  Running NEW optimization...")
        new_sol, new_time = run_optimization(
            GeneticOptimization, data, start_hour, ngen, fixed_seed, config
        )
        print(f"  NEW done in {new_time:.2f}s")

        # Timing comparison
        speedup = old_time / new_time if new_time > 0 else 0
        pct = (speedup - 1) * 100
        print(f"\n  Time: OLD={old_time:.2f}s  NEW={new_time:.2f}s  "
              f"{'speedup' if speedup > 1 else 'slowdown'}: {max(speedup, 1/speedup):.2f}x ({pct:+.1f}%)")

        # Compare key metrics
        print(f"\n  Key Metrics:")
        print(f"    Gesamtbilanz_Euro:  OLD={old_sol.result.Gesamtbilanz_Euro:.6f}  "
              f"NEW={new_sol.result.Gesamtbilanz_Euro:.6f}  "
              f"diff={abs(old_sol.result.Gesamtbilanz_Euro - new_sol.result.Gesamtbilanz_Euro):.2e}")
        print(f"    Gesamtkosten_Euro:  OLD={old_sol.result.Gesamtkosten_Euro:.6f}  "
              f"NEW={new_sol.result.Gesamtkosten_Euro:.6f}")
        print(f"    Gesamteinnahmen_Euro: OLD={old_sol.result.Gesamteinnahmen_Euro:.6f}  "
              f"NEW={new_sol.result.Gesamteinnahmen_Euro:.6f}")
        print(f"    Gesamt_Verluste:    OLD={old_sol.result.Gesamt_Verluste:.2f}  "
              f"NEW={new_sol.result.Gesamt_Verluste:.2f}")

        # Full comparison
        issues = compare_solutions(old_sol, new_sol, inp.name)
        if issues:
            print(f"\n  Differences ({len(issues)}):")
            for i in issues:
                print(i)
        else:
            print(f"\n  IDENTICAL - No differences detected")

    print(f"\n{'=' * 70}")
    print("Full optimization benchmark complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
