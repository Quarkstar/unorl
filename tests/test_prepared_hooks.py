"""Actual backward collection across both Torch checkpoint implementations."""

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unorl.prepared_hooks import PreparedProjectionHook
from unorl.prepared_moments import PreparedBasisMoments


class LinearAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(91)
        self.base = torch.nn.Parameter(
            torch.randn(13, 11, dtype=torch.float64, generator=generator), requires_grad=False
        )
        self.a = torch.nn.Parameter(torch.randn(1, 11, dtype=torch.float64, generator=generator))
        self.b = torch.nn.Parameter(torch.randn(13, 1, dtype=torch.float64, generator=generator))

    def forward(self, x):
        return F.linear(x, self.base) + 4 * F.linear(F.linear(x, self.a), self.b)


def make_observer(layer):
    generator = torch.Generator().manual_seed(92)

    def basis(direction):
        first = direction.reshape(-1, 1) / direction.norm()
        second = torch.randn(first.shape, dtype=first.dtype, generator=generator)
        second -= first * (first.T @ second)
        return torch.cat((first, second / second.norm()), dim=1)

    return PreparedBasisMoments(basis(layer.a.detach()), basis(layer.b.detach()), chunk_tokens=2)


@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_hook_matches_native_gradients_once_with_checkpointing(checkpoint_mode):
    layer = LinearAdapter()
    observer = make_observer(layer)
    hook = PreparedProjectionHook(layer, observer)
    x = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    output = (
        layer(x) if checkpoint_mode is None else checkpoint(layer, x, use_reentrant=checkpoint_mode)
    )
    output.square().mean().backward()
    assert hook.backward_calls == observer.pending_microbatches == 1
    assert layer.base.grad is None
    ac = observer.qin.new_tensor([[layer.a.norm().item(), 0.0]])
    bc = observer.qout.new_tensor([[layer.b.norm().item()], [0.0]])
    observer.finish_update(1)
    mapped = observer.map_moments(ac, bc, 4)
    assert torch.allclose(mapped["ma"] / (1 - observer.beta1), layer.a.grad, atol=1e-10, rtol=1e-12)
    assert torch.allclose(mapped["mb"] / (1 - observer.beta1), layer.b.grad, atol=1e-10, rtol=1e-12)
    assert torch.allclose(
        mapped["va"] / (1 - observer.beta2), layer.a.grad.square(), atol=1e-10, rtol=1e-12
    )
    assert torch.allclose(
        mapped["vb"] / (1 - observer.beta2), layer.b.grad.square(), atol=1e-10, rtol=1e-12
    )
    hook.close()


def test_hook_skips_eval_and_closing_disables_already_created_graph():
    layer = LinearAdapter()
    observer = make_observer(layer)
    hook = PreparedProjectionHook(layer, observer)
    with torch.no_grad():
        layer(torch.ones(2, 11, dtype=torch.float64))
    output = layer(torch.ones(2, 11, dtype=torch.float64))
    hook.close()
    output.sum().backward()
    assert hook.backward_calls == observer.pending_microbatches == 0


def test_hook_releases_activation_after_backward_even_with_output_retained():
    layer = LinearAdapter()
    observer = make_observer(layer)
    captured_inputs = []
    accumulate = observer.accumulate

    def record_input(inputs, gradient):
        # This is the detached tensor held by the gradient callback, not only
        # the caller's original wrapper around its shared activation storage.
        captured_inputs.append(weakref.ref(inputs))
        accumulate(inputs, gradient)

    observer.accumulate = record_input
    hook = PreparedProjectionHook(layer, observer)
    x = torch.randn(3, 11, dtype=torch.float64)
    reference = weakref.ref(x)
    output = layer(x)
    output.sum().backward()
    del x
    gc.collect()
    assert reference() is None
    assert len(captured_inputs) == 1 and captured_inputs[0]() is None
    assert hook.backward_calls == 1
    assert output.shape == (3, 13)
    hook.close()


def test_discarded_nonfinite_update_does_not_advance_history():
    layer = LinearAdapter()
    observer = make_observer(layer)
    observer.accumulate(
        torch.ones(2, 11, dtype=torch.float64),
        torch.full((2, 13), float("nan"), dtype=torch.float64),
    )
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        observer.finish_update(1)
    observer.discard_update()
    assert observer.count == observer.pending_microbatches == 0
    assert observer.ga.count_nonzero() == observer.gb.count_nonzero() == 0
    assert observer.ma.count_nonzero() == observer.mb.count_nonzero() == 0
