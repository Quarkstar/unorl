"""Prepared state agrees with independently replayed dense-gradient Adam."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unorl.prepared_moments import PreparedBasisMoments


class NoDenseGradient(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            if result.shape == (13, 11):
                raise AssertionError("Collector formed a dense weight-gradient product")
        return result


def observer(dtype=torch.float64):
    generator = torch.Generator().manual_seed(83)
    qin = torch.linalg.qr(torch.randn(11, 2, dtype=dtype, generator=generator)).Q
    qout = torch.linalg.qr(torch.randn(13, 2, dtype=dtype, generator=generator)).Q
    return PreparedBasisMoments(qin, qout, chunk_tokens=2), generator


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_chunked_collector_matches_native_adam_after_adaptive_selection(dtype, tolerance):
    state, generator = observer(dtype)
    history = []
    for update in range(8):
        dense = torch.zeros(13, 11, dtype=dtype)
        for shape in ((2, 11), (1, 5, 11), (3, 11)):
            x = torch.randn(*shape, dtype=dtype, generator=generator)
            dy = torch.randn(*shape[:-1], 13, dtype=dtype, generator=generator) / 10
            with NoDenseGradient():
                state.accumulate(x, dy)
            dense += dy.reshape(-1, 13).T @ x.reshape(-1, 11)
        clip = 0.2 + 0.05 * update
        history.append(dense * clip)
        state.finish_update(clip)
    ac, bc = state.select_coefficients(0.7, 0.3)
    mapped = state.map_moments(ac, bc, 4)
    parameters = [torch.nn.Parameter(torch.zeros_like(mapped[key])) for key in ("a", "b")]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0)
    for gradient in history:
        parameters[0].grad = 4 * mapped["b"].T @ gradient
        parameters[1].grad = 4 * gradient @ mapped["a"].T
        optimizer.step()
    for parameter, first, second in zip(parameters, ("ma", "mb"), ("va", "vb")):
        assert torch.allclose(
            optimizer.state[parameter]["exp_avg"], mapped[first], atol=tolerance, rtol=tolerance
        )
        assert torch.allclose(
            optimizer.state[parameter]["exp_avg_sq"], mapped[second], atol=tolerance, rtol=tolerance
        )
        assert optimizer.state[parameter]["step"].item() == mapped["step"] == 8


def test_resume_preserves_unfinished_microbatch_and_does_not_alias():
    state, generator = observer()
    x = torch.randn(5, 11, dtype=torch.float64, generator=generator)
    dy = torch.randn(5, 13, dtype=torch.float64, generator=generator)
    state.accumulate(x, dy)
    checkpoint = state.state_dict()
    resumed = PreparedBasisMoments.from_state_dict(checkpoint)
    resumed.accumulate(x, dy)
    state.accumulate(x, dy)
    for current in (state, resumed):
        current.finish_update(0.4)
    assert resumed.count == state.count == 1
    assert checkpoint["count"] == 0 and checkpoint["pending_microbatches"] == 1
    for name in state._buffers:
        assert torch.equal(getattr(state, name), getattr(resumed, name))
    checkpoint["ga"].zero_()
    assert resumed.ma.count_nonzero() > 0


def test_cross_moment_is_required_for_combined_directions():
    q = torch.eye(2, dtype=torch.float64)
    state = PreparedBasisMoments(q, q)
    state.accumulate(torch.ones(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64))
    state.finish_update(1)
    coefficients = torch.ones(1, 2, dtype=torch.float64)
    mapped = state.map_moments(coefficients, coefficients.T, 1)
    # Native gradient per output is 2: its squared value is 4, not 1+1.
    assert torch.allclose(mapped["vb"], torch.full((2, 1), 0.004, dtype=torch.float64))
    assert not torch.allclose(mapped["vb"], torch.full((2, 1), 0.002, dtype=torch.float64))


def test_noncontiguous_microbatch_and_corrupt_checkpoint():
    state, generator = observer()
    with pytest.raises(ValueError, match="without any microbatch"):
        state.finish_update(1)
    x = torch.randn(2, 3, 11, dtype=torch.float64, generator=generator).transpose(0, 1)
    dy = torch.randn(3, 2, 13, dtype=torch.float64, generator=generator)
    with NoDenseGradient():
        state.accumulate(x, dy)
    dense = dy.reshape(-1, 13).T @ x.reshape(-1, 11)
    assert torch.allclose(state.ga, state.qout.T @ dense, atol=1e-12, rtol=0)
    assert torch.allclose(state.gb, dense @ state.qin, atol=1e-12, rtol=0)
    state.finish_update(1)
    checkpoint = state.state_dict()
    checkpoint["ca"][0, 0] = -1
    with pytest.raises(ValueError, match="Negative diagonal"):
        PreparedBasisMoments.from_state_dict(checkpoint)


def test_rejects_covariance_without_valid_cross_moments():
    state, _ = observer()
    checkpoint = state.state_dict()
    checkpoint["count"] = 1
    checkpoint["ca"][:, 0] = torch.tensor([1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="positive semidefinite"):
        PreparedBasisMoments.from_state_dict(checkpoint)


def test_dense_product_guard_has_a_negative_control():
    with pytest.raises(AssertionError, match="dense weight-gradient"):
        with NoDenseGradient():
            torch.zeros(13, 2) @ torch.zeros(2, 11)


def test_observer_storage_is_fixed_with_observation_count():
    state, _ = observer(torch.float32)
    before = state.resident_tensor_bytes
    assert before == 9 * (11 + 13) * 4
    for _ in range(3):
        state.accumulate(torch.ones(1, 11), torch.ones(1, 13))
        state.finish_update(1)
    assert state.resident_tensor_bytes == before
