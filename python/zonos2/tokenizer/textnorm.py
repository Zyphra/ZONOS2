"""TTS text normalization using the vendored NeMo forward-TN grammars.

Converts written text to its spoken form ("$5.32" -> "five dollars thirty two
cents") before tokenization. Grammar construction is expensive (~10-30s per
language) the first time; compiled WFSTs are cached as .far files under
ZONOS2_TTS_NORM_CACHE_DIR (default ~/.cache/zonos2-tts-norm) and reload in
well under a second. Use tools/build_tts_norm_fars.py to prewarm all caches.

Disable globally with ZONOS2_TTS_NORM=0, or per request via the
text_normalization flag.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Dict, Optional

from zonos2.utils import init_logger

logger = init_logger(__name__)

# Server language codes -> NeMo text_normalization language packages.
SERVER_TO_NEMO_LANG: Dict[str, str] = {
    "en_us": "en",
    "en_gb": "en",
    "fr_fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt_br": "pt",
    "ja": "ja",
    "cmn": "zh",
    "ko": "ko",
}

# Upstream's own tests run Korean grammars with lower_cased input; everything
# else uses cased.
LOWER_CASED_LANGS = {"ko"}

# zh/ja verbalizers read a cached .far but never write one (upstream quirk);
# we post-write it after first compile so later loads are fast. Note ja reads
# a "jp_" prefixed name.
_VERBALIZER_FAR_PREFIX = {"zh": "zh", "ja": "jp"}

# A digit directly followed by sentence punctuation confuses several upstream
# r1.2.0 grammars: pt's tagger raises FstOpError outright and de reads dates
# digit-by-digit (reproduced with upstream code and its pinned pynini
# 2.1.6.post1 — upstream bugs, not vendoring artifacts). Space the punctuation
# off before normalization; it is re-attached afterwards.
_DIGIT_PUNCT_RE = re.compile(r"(\d)([.!?,;:])(?=\s|$)")
_SPACE_PUNCT_RE = re.compile(r" +([.!?,;:])(?=\s|$)")

# Moses-based punct_post_process re-attaches punctuation well for these
# languages. For the European languages it also glues currency symbols to the
# following word ("5,32 € am" -> "€am"), so there we skip moses and only
# collapse the spacing we introduced ourselves.
_MOSES_POSTPROCESS_LANGS = {"en", "zh", "ja", "ko"}


def default_cache_root() -> str:
    return os.environ.get(
        "ZONOS2_TTS_NORM_CACHE_DIR",
        os.path.expanduser("~/.cache/zonos2-tts-norm"),
    )


def normalization_enabled() -> bool:
    return os.environ.get("ZONOS2_TTS_NORM", "1") != "0"


class TTSTextNormalizer:
    """Lazy per-language NeMo normalizers with .far caching.

    Construction and calls are serialized per language: the upstream
    Normalizer shares a mutable TokenParser and is not thread-safe.
    """

    def __init__(self, cache_root: str | None = None):
        self.cache_root = cache_root or default_cache_root()
        self._normalizers: Dict[str, object] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @staticmethod
    def nemo_lang(language: str) -> Optional[str]:
        return SERVER_TO_NEMO_LANG.get(language)

    def supported(self, language: str) -> bool:
        return language in SERVER_TO_NEMO_LANG

    def _lang_lock(self, lang: str) -> threading.Lock:
        with self._global_lock:
            if lang not in self._locks:
                self._locks[lang] = threading.Lock()
            return self._locks[lang]

    def _build(self, lang: str):
        from zonos2.vendor.nemo_text_processing.text_normalization import (
            Normalizer,
        )

        input_case = "lower_cased" if lang in LOWER_CASED_LANGS else "cased"
        # One cache dir per (lang, case): upstream .far filenames collide
        # across languages (e.g. ja's tagger writes a zh_-prefixed file).
        cache_dir = os.path.join(self.cache_root, f"{lang}_{input_case}")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading TTS text normalizer for '%s' (%s)...", lang, input_case)
        normalizer = Normalizer(
            input_case=input_case,
            lang=lang,
            cache_dir=cache_dir,
            overwrite_cache=False,
        )

        prefix = _VERBALIZER_FAR_PREFIX.get(lang)
        if prefix is not None:
            far_path = os.path.join(
                cache_dir, f"{prefix}_tn_True_deterministic_verbalizer.far"
            )
            if not os.path.exists(far_path):
                from zonos2.vendor.nemo_text_processing.text_normalization.en.graph_utils import (
                    generator_main,
                )

                generator_main(far_path, {"verbalize": normalizer.verbalizer.fst})
        return normalizer

    def get(self, lang: str):
        with self._lang_lock(lang):
            if lang not in self._normalizers:
                self._normalizers[lang] = self._build(lang)
            return self._normalizers[lang]

    def warmup(self, languages: list[str] | None = None) -> None:
        """Construct normalizers ahead of time (server codes or NeMo codes)."""
        langs = languages or sorted(set(SERVER_TO_NEMO_LANG.values()))
        for lang in langs:
            lang = SERVER_TO_NEMO_LANG.get(lang, lang)
            try:
                self.get(lang)
            except Exception:  # noqa: BLE001
                logger.exception("TTS text normalizer warmup failed for '%s'", lang)

    def normalize(self, text: str, language: str) -> str:
        """Normalize text for the given server language code.

        Returns the input unchanged for unsupported languages or on any
        normalizer error — normalization must never fail a request.
        """
        lang = SERVER_TO_NEMO_LANG.get(language)
        if lang is None or not text.strip():
            return text
        text_in = _DIGIT_PUNCT_RE.sub(r"\1 \2", text)
        use_moses = lang in _MOSES_POSTPROCESS_LANGS
        try:
            normalizer = self.get(lang)
            with self._lang_lock(lang):
                result = normalizer.normalize(text_in, punct_post_process=use_moses)
        except Exception:  # noqa: BLE001
            logger.exception(
                "TTS text normalization failed for lang=%s; using raw text", language
            )
            return text
        if isinstance(result, str):
            result = _SPACE_PUNCT_RE.sub(r"\1", result)
        if not isinstance(result, str) or not result.strip():
            return text
        logger.debug("TTS norm [%s]: %r -> %r", language, text, result)
        return result
