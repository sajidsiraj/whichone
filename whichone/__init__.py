"""whichone: two-stage MODM+MADM decision support for LLM selection in agentic AI.

Layers:
  whichone.core       pure decision engine (no web or storage dependencies)
  whichone.catalogue  marketplace data layer (versioned snapshots)
  whichone.api        FastAPI service exposing the engine (optional extra)
"""

__version__ = "0.1.0"
