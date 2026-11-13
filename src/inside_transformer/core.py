"""Metrics and explicit TransformerLens run_with_hooks interventions."""
import torch


def logit_difference(logits, answers, foils):
    last = logits[:, -1, :]
    ids = torch.arange(last.shape[0], device=last.device)
    return last[ids, answers] - last[ids, foils]


def replacement_hook(donor, position):
    def hook(activation, hook):
        activation[:, position, :] = donor[:, position, :]
        return activation
    return hook


def patched_logits(model, tokens, donor_cache, layer, position):
    name = f"blocks.{layer}.hook_resid_pre"
    return model.run_with_hooks(tokens, fwd_hooks=[(name, replacement_hook(donor_cache[name], position))], return_type="logits")


def residual_filter(name):
    return name.endswith("hook_resid_pre")
