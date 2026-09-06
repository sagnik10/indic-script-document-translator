"""AI-assisted multilingual document translation package."""

from .config.settings import Settings, get_settings
from .pipeline import DocumentTranslationPipeline

__all__ = ["DocumentTranslationPipeline", "Settings", "get_settings"]
__version__ = "0.1.0"

