import asyncio

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
api_server = pytest.importorskip("zonos2.server.api_server")


class _DummyModelConfig:
    speaker_enabled = True
    speaker_embedding_dim = 3


class _DummyConfig:
    model_config = _DummyModelConfig()
    tts_default_voices_dir = None


class _DefaultVoiceConfig(_DummyConfig):
    def __init__(self, default_voices_dir):
        self.tts_default_voices_dir = str(default_voices_dir)


def test_slerp_embeddings_respects_endpoints_and_midpoint():
    v0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    start = api_server._slerp_embeddings(v0, v1, 0.0)
    end = api_server._slerp_embeddings(v0, v1, 1.0)
    mid = api_server._slerp_embeddings(v0, v1, 0.5)

    assert torch.allclose(start, v0)
    assert torch.allclose(end, v1)
    expected_mid = torch.tensor([2**-0.5, 2**-0.5, 0.0], dtype=torch.float32)
    assert torch.allclose(mid, expected_mid, atol=1e-5)


def test_resolve_speaker_embedding_uses_cached_ids_and_blends():
    async def _run():
        api_server._SESSION_SPEAKER_CACHE.clear()
        speaker_a = await api_server._cache_speaker_reference(
            session_id="test-session",
            label="Speaker A",
            source_type="audio",
            embedding=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        )
        speaker_b = await api_server._cache_speaker_reference(
            session_id="test-session",
            label="Speaker B",
            source_type="audio",
            embedding=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        )

        single = await api_server._resolve_speaker_embedding(
            config=_DummyConfig(),
            session_id="test-session",
            speaker_embedding_id=speaker_a.speaker_id,
        )
        blended = await api_server._resolve_speaker_embedding(
            config=_DummyConfig(),
            session_id="test-session",
            speaker_blend_embedding_id_a=speaker_a.speaker_id,
            speaker_blend_embedding_id_b=speaker_b.speaker_id,
            speaker_blend_t=0.5,
        )
        return single, blended

    single, blended = asyncio.run(_run())

    assert torch.allclose(single, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    expected_mid = torch.tensor([2**-0.5, 2**-0.5, 0.0], dtype=torch.float32)
    assert torch.allclose(blended, expected_mid, atol=1e-5)


def test_default_speaker_directory_lists_and_resolves_embedding(tmp_path):
    async def _run():
        api_server._DEFAULT_SPEAKER_CACHE.clear()
        np.save(tmp_path / "Alice Voice.npy", np.array([0.25, 0.5, 0.75], dtype=np.float32))

        config = _DefaultVoiceConfig(tmp_path)
        speakers = await api_server._list_default_speakers(config)
        resolved = await api_server._resolve_speaker_embedding(
            config=config,
            session_id=None,
            speaker_embedding_id=speakers[0].speaker_id,
        )
        serialized = api_server._serialize_default_speaker(speakers[0], expected_dim=3)
        return speakers, resolved, serialized

    speakers, resolved, serialized = asyncio.run(_run())

    assert len(speakers) == 1
    assert speakers[0].label == "Alice Voice"
    assert speakers[0].source_type == "embedding_file"
    assert torch.allclose(resolved, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32))
    assert serialized["is_default"] is True
    assert serialized["has_preview"] is False
