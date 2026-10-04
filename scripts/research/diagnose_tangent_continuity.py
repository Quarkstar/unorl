"""Check the geometric lower bound on rank-one update transport using NumPy.

This is a CPU linear-algebra diagnostic, not an Adam or RL performance test.
The least-squares variables allow arbitrary factor directions; native Adam
with transported state is more constrained and need not reach this bound.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def normalize(vector):
    return vector / np.linalg.norm(vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for seed in (17, 29, 43, 71):
        rng = np.random.default_rng(seed)
        u, v = normalize(rng.normal(size=7)), normalize(rng.normal(size=11))
        u_extra = rng.normal(size=7)
        u_extra = normalize(u_extra - u * (u @ u_extra))
        v_extra = rng.normal(size=11)
        v_extra = normalize(v_extra - v * (v @ v_extra))
        old_direction = np.outer(u, rng.normal(size=11)) + np.outer(rng.normal(size=7), v)
        for angle_u, angle_v in ((0.0, 0.0), (0.2, 0.2), (0.6, 0.6), (0.6, 0.0)):
            un = np.cos(angle_u) * u + np.sin(angle_u) * u_extra
            vn = np.cos(angle_v) * v + np.sin(angle_v) * v_extra
            pu, pv = np.outer(un, un), np.outer(vn, vn)
            projection = pu @ old_direction + old_direction @ pv - pu @ old_direction @ pv
            normal = (np.eye(7) - pu) @ old_direction @ (np.eye(11) - pv)
            columns = [np.outer(un, axis).ravel() for axis in np.eye(11)]
            columns += [np.outer(axis, vn).ravel() for axis in np.eye(7)]
            design = np.column_stack(columns)
            coefficients = np.linalg.lstsq(design, old_direction.ravel(), rcond=None)[0]
            fitted = (design @ coefficients).reshape(7, 11)
            error = float(np.linalg.norm(fitted - projection))
            residual_error = float(
                abs(np.linalg.norm(old_direction - fitted) - np.linalg.norm(normal))
            )
            if error > 1e-12 or residual_error > 1e-12:
                raise AssertionError("Tangent projection differs from unrestricted least squares")
            cases.append(
                {
                    "seed": seed,
                    "output_angle_radians": angle_u,
                    "input_angle_radians": angle_v,
                    "minimum_relative_update_error": float(
                        np.linalg.norm(normal) / np.linalg.norm(old_direction)
                    ),
                    "projection_vs_lstsq_error": error,
                    "residual_identity_error": residual_error,
                }
            )
    result = {
        "scope": "Geometric tangent lower bound; unrestricted factor directions, not native Adam or RL",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {len(cases)} tangent-projection checks; wrote {args.output}")


if __name__ == "__main__":
    main()
