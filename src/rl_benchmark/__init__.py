"""RL benchmark package."""

from pathlib import Path

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "__version__"]

__version__ = "0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
