"""Single-use output-gradient hooks for prepared linear-layer projections."""

import torch


class PreparedProjectionHook:
    """Observe training backward calls; skip evaluation and no-grad recomputation.

    Activation references live in the autograd callback, never in the observer.
    The callback removes itself after use. Closing disables callbacks from
    already-created graphs as well as future forward registration.
    """

    def __init__(self, layer, observer):
        self.observer = observer
        self.enabled = True
        self.backward_calls = 0
        self._forward_handle = layer.register_forward_hook(self._forward)

    def _forward(self, layer, args, output):
        if not self.enabled or not torch.is_grad_enabled():
            return
        if (
            not isinstance(output, torch.Tensor)
            or not args
            or not isinstance(args[0], torch.Tensor)
        ):
            raise TypeError("Prepared projections require a tensor-in/tensor-out linear layer")
        if not output.requires_grad:
            return
        inputs = args[0].detach()

        def capture(gradient):
            try:
                if self.enabled:
                    self.observer.accumulate(inputs, gradient)
                    self.backward_calls += 1
            finally:
                handle.remove()

        handle = output.register_hook(capture)

    def close(self):
        self.enabled = False
        self._forward_handle.remove()
