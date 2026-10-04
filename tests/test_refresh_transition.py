"""Validate gradual rotation with intervening updates and saved transition state."""

import math

import pytest
import torch

from unorl.refresh_transition import make_refresh_plane, rotate_in_refresh_plane


def test_composition_and_off_plane_preservation():
    a = torch.randn(1, 13, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    plane = make_refresh_plane(a, seed=17)
    row = a.clone()
    for _ in range(10):
        row = rotate_in_refresh_plane(row, plane, 2)
    torch.testing.assert_close(row, rotate_in_refresh_plane(a, plane, 20))
    trained_a = a + torch.arange(13, dtype=a.dtype).reshape(1, -1) * 0.01
    rotated = rotate_in_refresh_plane(trained_a, plane, 2)

    def off_plane(value):
        return value - sum((value * unit).sum() * unit for unit in plane)

    torch.testing.assert_close(off_plane(rotated), off_plane(trained_a))
    torch.testing.assert_close(rotated.norm(), trained_a.norm())
    cosine = (rotated * trained_a).sum() / (rotated.norm() * trained_a.norm())
    assert cosine >= math.cos(math.radians(2)) - 1e-12


def test_compensation_and_resume_with_intervening_adam_updates(tmp_path):
    generator = torch.Generator().manual_seed(42)
    a = torch.nn.Parameter(torch.randn(1, 13, generator=generator, dtype=torch.float64))
    b = torch.nn.Parameter(torch.randn(11, 1, generator=generator, dtype=torch.float64))
    base = torch.randn(11, 13, generator=generator, dtype=torch.float64)
    target = torch.randn_like(base)
    scale = 32
    optimizer = torch.optim.AdamW([a, b], lr=0.001, weight_decay=0)
    plane = make_refresh_plane(a, seed=17)

    def update_then_rotate(row, column, weight, opt, basis):
        opt.zero_grad()
        (weight + scale * column @ row - target).square().mean().backward()
        opt.step()
        before = weight + scale * column @ row
        with torch.no_grad():
            fresh = rotate_in_refresh_plane(row, basis, 2)
            weight.add_(scale * column @ (row - fresh))
            row.copy_(fresh)
        torch.testing.assert_close(weight + scale * column @ row, before)

    for _ in range(3):
        update_then_rotate(a, b, base, optimizer, plane)
    checkpoint = tmp_path / "transition.pt"
    torch.save(
        {
            "a": a.detach(),
            "b": b.detach(),
            "base": base,
            "optimizer": optimizer.state_dict(),
            "plane": plane,
            "increments_complete": 3,
        },
        checkpoint,
    )
    for _ in range(7):
        update_then_rotate(a, b, base, optimizer, plane)
    saved = torch.load(checkpoint, weights_only=True)
    restored_a = torch.nn.Parameter(saved["a"])
    restored_b = torch.nn.Parameter(saved["b"])
    restored_optimizer = torch.optim.AdamW([restored_a, restored_b], lr=0.001, weight_decay=0)
    restored_optimizer.load_state_dict(saved["optimizer"])
    for _ in range(saved["increments_complete"], 10):
        update_then_rotate(
            restored_a, restored_b, saved["base"], restored_optimizer, saved["plane"]
        )
    for actual, restored in ((a, restored_a), (b, restored_b), (base, saved["base"])):
        torch.testing.assert_close(actual, restored, atol=0, rtol=0)
    assert optimizer.state[a]["step"] == restored_optimizer.state[restored_a]["step"] == 10


def test_invalid_transition_inputs():
    a = torch.ones(1, 4, dtype=torch.float64)
    plane = make_refresh_plane(a, seed=17)
    for angle in (-1, 90, float("nan")):
        with pytest.raises(ValueError):
            rotate_in_refresh_plane(a, plane, angle)
    with pytest.raises(ValueError):
        make_refresh_plane(torch.zeros_like(a), seed=17)
    with pytest.raises(ValueError):
        rotate_in_refresh_plane(a, (plane[0], plane[0]), 2)
