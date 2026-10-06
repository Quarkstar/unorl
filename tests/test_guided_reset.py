import pytest
import torch
from torch.utils.checkpoint import checkpoint

from unorl.guided_reset import GradientSketch, choose_direction, fresh_probes, update_spaces
from unorl.prepared_hooks import PreparedProjectionHook


def test_spaces_account_for_cancelled_history():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[2.0], [0.0], [0.0]])
    u, v, values = update_spaces([(a, b), (a, -b)])
    assert u.shape == v.shape == (3, 0)
    assert torch.equal(values, torch.zeros_like(values))


def test_microbatch_sketch_matches_dense_gradient():
    torch.manual_seed(42)
    x, dy = torch.randn(11, 4), torch.randn(11, 5)
    probes = torch.linalg.qr(torch.randn(4, 2)).Q
    sketch = GradientSketch(probes, 5, chunk_tokens=3)
    sketch.accumulate(x[:4], dy[:4])
    sketch.accumulate(x[4:], dy[4:])
    torch.testing.assert_close(sketch.gradient, (dy.T @ x) @ probes)


def test_guided_reset_finds_unused_descent_direction():
    a, b = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[2.0], [0.0], [0.0]])
    u, v, _ = update_spaces([(a, b)])
    probes = fresh_probes(v, 3, 2, seed=42, device="cpu")
    sketch = GradientSketch(probes, 3)
    gradient = torch.diag(torch.tensor([8.0, 3.0, 1.0]))
    sketch.gradient.copy_(gradient @ probes)
    new_a, metrics = choose_direction(sketch, u, a * 0.2)
    torch.testing.assert_close(new_a.norm(), torch.tensor(0.2))
    torch.testing.assert_close(new_a.abs(), torch.tensor([[0.0, 0.2, 0.0]]), atol=1e-6, rtol=0)
    assert not metrics["fallback_to_random"]
    torch.testing.assert_close(v.T @ new_a.T, torch.zeros(1, 1), atol=1e-6, rtol=0)


def test_zero_normal_signal_is_explicit_random_fallback():
    probes = torch.eye(3)[:, 1:]
    sketch = GradientSketch(probes, 3)
    sketch.gradient[0].fill_(1)
    reference = torch.tensor([[0.1, 0.2, 0.3]])
    direction, metrics = choose_direction(sketch, torch.eye(3)[:, :1], reference)
    assert metrics["fallback_to_random"]
    assert torch.equal(direction, reference)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_sketch_matches_native_rank_one_b_gradient(checkpointed):
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("base", torch.randn(5, 4))
            self.a = torch.nn.Parameter(torch.randn(1, 4))
            self.b = torch.nn.Parameter(torch.zeros(5, 1))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.base + 32 * (self.b @ self.a))

    torch.manual_seed(91)
    layer = Adapter()
    probes = torch.eye(4)
    sketch = GradientSketch(probes, 5, chunk_tokens=2)
    hook = PreparedProjectionHook(layer, sketch)
    x = torch.randn(2, 3, 4, requires_grad=True)
    output = checkpoint(layer, x, use_reentrant=False) if checkpointed else layer(x)
    output.square().mean().backward()
    assert sketch.observations == 1
    torch.testing.assert_close(layer.b.grad, 32 * (sketch.gradient @ layer.a.detach().T))
    hook.close()
