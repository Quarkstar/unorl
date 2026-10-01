"""Compare the rank-one specialization with the pinned authors' LoFT-simple code."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

PIN = "148998d98f901cd7744db7baa5b2b4868fa16bf0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PIN:
        raise ValueError(f"LoFT reference revision mismatch: {actual}")
    sys.path.insert(0, str(args.reference.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from loft_optim.clipping import clip_grad_norm
    from loft_optim.optimizer import LoFTAdamW
    from test_loft import tiny_model

    from unorl.loft import LoFTSimpleAdamW

    model = tiny_model()
    reference = copy.deepcopy(model)
    trial = LoFTSimpleAdamW(model, lr=0.001)
    author = LoFTAdamW(
        reference.parameters(),
        lr=0.001,
        eps=1e-4,
        weight_decay=0.0,
        model=reference,
        reproject_second_moment=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    maximum = 0.0
    for _ in range(8):
        for current in [model, reference]:
            current(tokens, labels=tokens).loss.backward()
        norm = trial.calibrated_grad_norm()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(min(1.0, 1.0 / (norm + 1e-6)))
        clip_grad_norm(reference, author, max_norm=1.0)
        trial.step()
        author.step()
        trial.zero_grad()
        author.zero_grad()
        for (name, p), (other_name, other) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            assert name == other_name
            maximum = max(maximum, (p - other).abs().max().item())
            torch.testing.assert_close(p, other, rtol=2e-5, atol=2e-7)
        assert trial.update_A == author.update_A
    result = {
        "success": True,
        "updates": 8,
        "maximum_parameter_difference": maximum,
        "reference_commit": PIN,
        "clipping_parity": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
