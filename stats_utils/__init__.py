"""Modern namespace for the stats-utils package.

The legacy top-level packages (`ab_testing_utils` and `quasi_experiment_utils`)
remain supported for backwards compatibility. New code should prefer imports
from this namespace.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stats-utils")
except PackageNotFoundError:  # pragma: no cover - only relevant in editable/source trees
    __version__ = "0.1.0"

__all__ = ["__version__"]
