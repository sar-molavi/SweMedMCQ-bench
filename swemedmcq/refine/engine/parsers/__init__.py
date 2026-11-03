"""vLLM reasoning-parser plugins (loaded by file path, never imported by the pool).

Importing these modules registers parsers with vLLM's ``ReasoningParserManager``;
vLLM does that itself from ``structured_outputs_config.reasoning_parser_plugin``.
vLLM's built-in parsers (e.g. ``gemma4`` for Gemma-4) have no entry in
``PLUGIN_PATHS``; the pool then sets no plugin path.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

GPTOSS_STRUCTURED = "gptoss_structured"
QWEN3_8_THINKING = "qwen3_8_thinking"

PLUGIN_PATHS: dict[str, Path] = {
    GPTOSS_STRUCTURED: _HERE / "gptoss_structured.py",
    QWEN3_8_THINKING: _HERE / "qwen3_8_thinking.py",
}
