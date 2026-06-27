import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
emotion = pytest.importorskip("zonos2.tts.emotion")


def _write_directions(tmp_path, dim=4):
    """Write a tiny directions set: 'happy' named + 'valence' axis."""
    happy = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    valence = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.save(tmp_path / "happy.npy", happy)
    np.save(tmp_path / "valence.npy", valence)
    manifest = {
        "dim": dim,
        "ref_base_norm": 10.0,
        "neutral": "neutral",
        "directions": {
            "happy": {"file": "happy.npy", "kind": "named", "norm": 1.0},
            "valence": {"file": "valence.npy", "kind": "axis", "norm": 1.0},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return emotion.EmotionDirections.load(tmp_path)


def test_load_classifies_named_and_axis(tmp_path):
    directions = _write_directions(tmp_path)
    assert directions is not None
    assert directions.dim == 4
    assert directions.emotion_names == ["happy"]
    assert directions.axis_names == ["valence"]
    assert not directions.is_empty()


def test_load_missing_returns_none(tmp_path):
    assert emotion.EmotionDirections.load(tmp_path / "nope") is None


def test_zero_request_is_identity(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float32)

    assert torch.allclose(emotion.apply_emotion(base, directions=directions), base)
    assert torch.allclose(
        emotion.apply_emotion(base, directions=directions, sliders={"happy": 0.0}), base
    )
    # No directions at all -> identity.
    assert torch.allclose(
        emotion.apply_emotion(base, directions=None, sliders={"happy": 1.0}), base
    )


def test_slider_moves_toward_direction_and_preserves_norm(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float32)  # norm 5
    base_norm = float(torch.linalg.vector_norm(base))

    out = emotion.apply_emotion(base, directions=directions, sliders={"happy": 1.0})
    # Norm preserved.
    assert float(torch.linalg.vector_norm(out)) == pytest.approx(base_norm, abs=1e-5)
    # Moved along +happy (x component grows relative to baseline direction).
    assert out[0] > base[0]


def test_no_preserve_norm_is_raw_sum(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.zeros(4, dtype=torch.float32)
    out = emotion.apply_emotion(
        base, directions=directions, sliders={"happy": 2.0}, strength=1.5, preserve_norm=False
    )
    # base + strength * (2 * happy) = 3.0 on x.
    assert out[0] == pytest.approx(3.0, abs=1e-6)


def test_valence_axis_and_linearity(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.zeros(4, dtype=torch.float32)
    out = emotion.apply_emotion(
        base, directions=directions, valence=0.5, strength=2.0, preserve_norm=False
    )
    assert out[1] == pytest.approx(1.0, abs=1e-6)  # 2.0 * 0.5 * valence


def test_unknown_slider_strict_raises(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        emotion.apply_emotion(base, directions=directions, sliders={"bogus": 1.0}, strict=True)
    # Non-strict skips it (identity since nothing else requested).
    out = emotion.apply_emotion(base, directions=directions, sliders={"bogus": 1.0}, strict=False)
    assert torch.allclose(out, base)


def test_dim_mismatch_raises(tmp_path):
    directions = _write_directions(tmp_path)
    base = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError):
        emotion.apply_emotion(base, directions=directions, sliders={"happy": 1.0})


def _write_lda_directions(tmp_path, lda_dim=3, input_dim=6):
    """Tiny LDA-space directions set with a random affine W x + b."""
    torch.manual_seed(0)
    W = torch.randn(lda_dim, input_dim)
    b = torch.randn(lda_dim)
    Wp = torch.linalg.pinv(W)
    happy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.save(tmp_path / "happy.npy", happy)
    np.save(tmp_path / "lda_weight.npy", W.numpy())
    np.save(tmp_path / "lda_bias.npy", b.numpy())
    np.save(tmp_path / "lda_pinv.npy", Wp.numpy())
    manifest = {
        "dim": lda_dim, "input_dim": input_dim, "space": "lda", "ref_base_norm": 1.0,
        "directions": {"happy": {"file": "happy.npy", "kind": "named", "norm": 1.0}},
        "lda": {"weight": "lda_weight.npy", "bias": "lda_bias.npy", "pinv": "lda_pinv.npy"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return emotion.EmotionDirections.load(tmp_path), W, b


def test_lda_load_and_input_dim(tmp_path):
    d, W, b = _write_lda_directions(tmp_path)
    assert d.space == "lda"
    assert d.dim == 3 and d.expected_input_dim == 6
    assert d.lda_weight is not None and d.lda_pinv is not None


def test_lda_injection_roundtrips_through_model_lda(tmp_path):
    """apply_emotion (lda) must produce a raw vector that, when the model
    re-applies W x + b, recovers base+delta renormalised in LDA space."""
    d, W, b = _write_lda_directions(tmp_path)
    base = torch.randn(6)
    strength = 2.0
    raw_out = emotion.apply_emotion(
        base, directions=d, sliders={"happy": 1.0}, strength=strength, preserve_norm=True)
    assert raw_out.numel() == 6  # fed to the model as a speaker embedding

    # what the (unchanged) model computes:
    recovered = W @ raw_out + b
    lda_base = W @ base + b
    intended = lda_base + strength * d.named["happy"]
    intended = intended * (torch.linalg.vector_norm(lda_base) / torch.linalg.vector_norm(intended))
    assert torch.allclose(recovered, intended, atol=1e-4)
    # emotion actually moved the consumed vector
    assert not torch.allclose(recovered, lda_base, atol=1e-3)


def test_lda_zero_request_is_identity(tmp_path):
    d, W, b = _write_lda_directions(tmp_path)
    base = torch.randn(6)
    assert torch.allclose(emotion.apply_emotion(base, directions=d), base.to(torch.float32))


def _write_proj_directions(tmp_path, hidden=5):
    happy = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.save(tmp_path / "happy.npy", happy)
    manifest = {
        "dim": hidden, "space": "proj", "ref_base_norm": 1.0,
        "directions": {"happy": {"file": "happy.npy", "kind": "named", "norm": 1.0}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return emotion.EmotionDirections.load(tmp_path)


def test_proj_apply_emotion_leaves_embedding_unchanged(tmp_path):
    """For proj directions the embedding is untouched (delta applied in model)."""
    d = _write_proj_directions(tmp_path)
    base = torch.randn(2048)  # arbitrary embedding dim; proj apply is a no-op
    assert torch.allclose(emotion.apply_emotion(base, directions=d, sliders={"happy": 1.0}), base)


def test_proj_hidden_delta(tmp_path):
    d = _write_proj_directions(tmp_path)
    # nothing requested -> None
    assert emotion.emotion_hidden_delta(d, sliders={"happy": 0.0}) is None
    delta = emotion.emotion_hidden_delta(d, sliders={"happy": 1.0}, strength=3.0)
    assert delta is not None and delta.numel() == 5
    assert torch.allclose(delta, torch.tensor([3.0, 0, 0, 0, 0]))
    # raw/lda directions yield no hidden delta
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = _write_directions(raw_dir)
    assert emotion.emotion_hidden_delta(raw, sliders={"happy": 1.0}) is None


def test_calibration_load_and_lookup(tmp_path):
    cal = {
        "global_default": 4.0,
        "default": {"happy": 5.0, "sad": 6.0},
        "by_speaker": {"spkA": {"happy": 3.0, "angry": 8.0}},
    }
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    c = emotion.EmotionCalibration.load(tmp_path)
    assert c is not None
    # per-speaker value wins
    assert c.strength("spkA", "happy") == 3.0
    assert c.strength("spkA", "angry") == 8.0
    # fall back to per-emotion default when speaker lacks the emotion
    assert c.strength("spkA", "sad") == 6.0
    # unknown speaker -> per-emotion default
    assert c.strength("spkZ", "happy") == 5.0
    # unknown speaker + emotion -> global default
    assert c.strength("spkZ", "fearful") == 4.0
    # no speaker key -> per-emotion default / global default
    assert c.strength(None, "happy") == 5.0
    assert c.strength(None, "whatever") == 4.0


def test_calibration_missing_returns_none(tmp_path):
    assert emotion.EmotionCalibration.load(tmp_path) is None
