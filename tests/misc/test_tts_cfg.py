"""Unit tests for speaker-embedding classifier-free guidance (CFG) plumbing.

These cover the logit-combination math and the cond/uncond pairing logic in
``TTSScheduler`` without loading a model. End-to-end audio verification (running
a real checkpoint with cfg_scale > 1) is a separate GPU smoke test.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from zonos2.scheduler.scheduler import TTSScheduler


def _scheduler() -> TTSScheduler:
    # The CFG helpers only call sibling methods on self; skip the heavy __init__.
    return object.__new__(TTSScheduler)


def _req(cfg_scale: float = 1.0, twin=None, is_uncond: bool = False):
    return SimpleNamespace(cfg_scale=cfg_scale, cfg_twin=twin, is_cfg_uncond=is_uncond)


def test_cfg_pairs_identifies_cond_uncond():
    sched = _scheduler()
    twin = _req(is_uncond=True)
    cond = _req(cfg_scale=2.0, twin=twin)
    other = _req()
    batch = SimpleNamespace(reqs=[cond, twin, other])

    assert sched._cfg_pairs(batch) == [(0, 1, 2.0)]


def test_cfg_pairs_skips_disabled_and_missing_twin():
    sched = _scheduler()
    # cfg_scale == 1.0 -> not a guided pair even with a twin set.
    twin = _req(is_uncond=True)
    cond_disabled = _req(cfg_scale=1.0, twin=twin)
    # twin not present in the batch (e.g. already finished) -> skipped.
    absent_twin = _req(is_uncond=True)
    cond_orphan = _req(cfg_scale=3.0, twin=absent_twin)
    batch = SimpleNamespace(reqs=[cond_disabled, twin, cond_orphan])

    assert sched._cfg_pairs(batch) == []


def test_apply_cfg_combines_only_cond_row():
    sched = _scheduler()
    twin = _req(is_uncond=True)
    cond = _req(cfg_scale=2.0, twin=twin)
    other = _req()
    batch = SimpleNamespace(reqs=[cond, twin, other])

    torch.manual_seed(0)
    logits = torch.randn(3, 4, 8)
    guided = sched._apply_cfg(logits.clone(), batch)

    scale = 2.0
    expected_cond = logits[1] + scale * (logits[0] - logits[1])
    assert torch.allclose(guided[0], expected_cond, atol=1e-5)
    # Uncond row and unrelated row are untouched.
    assert torch.allclose(guided[1], logits[1], atol=1e-6)
    assert torch.allclose(guided[2], logits[2], atol=1e-6)


def test_apply_cfg_noop_without_pairs():
    sched = _scheduler()
    batch = SimpleNamespace(reqs=[_req(), _req()])
    logits = torch.randn(2, 4, 8)
    # No guided pairs -> returns the same tensor object unchanged.
    assert sched._apply_cfg(logits, batch) is logits


def test_apply_cfg_scale_one_is_identity_on_cond():
    sched = _scheduler()
    # Even if a twin is present, scale == 1.0 leaves the cond row equal to cond.
    twin = _req(is_uncond=True)
    cond = _req(cfg_scale=1.0, twin=twin)
    batch = SimpleNamespace(reqs=[cond, twin])
    logits = torch.randn(2, 4, 8)
    out = sched._apply_cfg(logits.clone(), batch)
    # scale==1 is filtered out entirely, so it's a no-op (same object).
    assert out is logits or torch.allclose(out[0], logits[0])


def test_apply_cfg_negative_scale_downweights_prefix():
    # Prefix CFG reuses the same combine path with cfg_scale set to
    # prefix_cfg_scale, which may be negative to oppose/downweight the prefix.
    sched = _scheduler()
    twin = _req(is_uncond=True)  # prefix-CFG twin keeps speaker; not read here.
    cond = _req(cfg_scale=-1.0, twin=twin)
    batch = SimpleNamespace(reqs=[cond, twin])

    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    guided = sched._apply_cfg(logits.clone(), batch)

    # guided = uncond + (-1)*(cond - uncond) = 2*uncond - cond
    expected = 2.0 * logits[1] - logits[0]
    assert torch.allclose(guided[0], expected, atol=1e-5)
    # Uncond row untouched.
    assert torch.allclose(guided[1], logits[1], atol=1e-6)


def test_apply_cfg_fractional_scale_blends():
    # prefix_cfg_scale=0.5 partially downweights toward the unconditional branch.
    sched = _scheduler()
    twin = _req(is_uncond=True)
    cond = _req(cfg_scale=0.5, twin=twin)
    batch = SimpleNamespace(reqs=[cond, twin])

    torch.manual_seed(1)
    logits = torch.randn(2, 4, 8)
    guided = sched._apply_cfg(logits.clone(), batch)

    expected = logits[1] + 0.5 * (logits[0] - logits[1])
    assert torch.allclose(guided[0], expected, atol=1e-5)


def test_sync_twin_tokens_mirrors_cond_onto_twin():
    sched = _scheduler()
    twin = _req(is_uncond=True)
    cond = _req(cfg_scale=2.0, twin=twin)
    other = _req()
    batch = SimpleNamespace(reqs=[cond, twin, other])

    next_tokens = torch.tensor(
        [[1, 2, 3], [9, 9, 9], [4, 5, 6]], dtype=torch.int32
    )
    sched._sync_cfg_twin_tokens(next_tokens, batch)

    # Twin row now equals the conditional row; others unchanged.
    assert torch.equal(next_tokens[1], next_tokens[0])
    assert torch.equal(next_tokens[2], torch.tensor([4, 5, 6], dtype=torch.int32))
