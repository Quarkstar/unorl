"""Exercise normal-descent selection in the native prepared-history fixture.

The patch substitutes only the fixture's selection call; production worker
behavior is unchanged. Its tiny adapters all have scale 32. The declared
0.9 local-descent bound is a validation setting, not an RL recipe decision.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts.check_prepared_moments_fsdp import main as transfer_fixture
from unorl.prepared_moments import PreparedBasisMoments
from unorl.prepared_selection import select_normal_descent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selections = []

    def choose(observer, a_norm, b_norm):
        selected = select_normal_descent(observer, a_norm, b_norm, 32, min_total_fraction=0.9)
        diagnostics = selected["diagnostics"]
        assert diagnostics["total_descent_retained_fraction"] >= 0.9 - 1e-6
        selections.append(diagnostics)
        return selected["a_coeff"], selected["b_coeff"]

    with patch.object(PreparedBasisMoments, "select_coefficients", choose):
        transfer_fixture()
    if int(os.environ["RANK"]) == 0:
        result_path = args.output / "result.json"
        result = json.loads(result_path.read_text())
        result["selector"] = "normal descent constrained to 90% total descent per factor"
        result["selection_scope"] = "tiny-fixture call substitution, not production integration"
        result["rank0_selection_diagnostics"] = selections
        result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
