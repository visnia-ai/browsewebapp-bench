"""BrowseWebApp bench."""

from .catalog import Catalog, load_catalog
from .runner import BenchmarkRunner
from .schema import TaskSpec

__all__ = ["BenchmarkRunner", "Catalog", "TaskSpec", "load_catalog"]
__version__ = "0.1.0"
