# Copyright (c) 2024 Jaerin Lee
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Grokfast EMA gradient filter.

Adapted from https://github.com/ironjr/grokfast under its MIT license. The update is
the repository's ``gradfilter_ema`` algorithm, extended to support multiple
named modules and checkpoint-restorable state with stable parameter names.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


@torch.no_grad()
def gradfilter_ema(
    modules: Mapping[str, nn.Module],
    grads: dict[str, torch.Tensor] | None = None,
    *,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Amplify slow gradients using Grokfast's exponential moving average."""
    previous = grads or {}
    updated = {}
    for module_name, module in modules.items():
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            name = f"{module_name}.{parameter_name}"
            gradient = parameter.grad.detach()
            if name in previous:
                average = previous[name].to(device=gradient.device, dtype=gradient.dtype)
                average.mul_(alpha).add_(gradient, alpha=1.0 - alpha)
            else:
                # This matches the official first-step behavior: the initial
                # EMA is the current gradient, so it is amplified immediately.
                average = gradient.clone()
            parameter.grad.add_(average, alpha=lamb)
            updated[name] = average
    return updated
