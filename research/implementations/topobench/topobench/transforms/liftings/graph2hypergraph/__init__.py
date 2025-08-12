"""Graph2HypergraphLifting module with automated exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any

from .base import Graph2HypergraphLifting


class ModuleExportsManager:
    """Manages automatic discovery and registration of Graph2Hypergraph lifting classes."""

    @staticmethod
    def is_lifting_class(obj: Any) -> bool:
        """Check if an object is a valid Graph2Hypergraph lifting class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid lifting class.

        Returns
        -------
        bool
            True if the object is a valid Graph2Hypergraph lifting class (non-private class
            inheriting from Graph2HypergraphLifting), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
            and issubclass(obj, Graph2HypergraphLifting)
            and obj != Graph2HypergraphLifting
        )

    @classmethod
    def discover_liftings(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all Graph2Hypergraph lifting classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        liftings = {}

        # Get the directory containing the lifting modules
        package_dir = Path(package_path).parent

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import the module
            module_name = f"{Path(package_path).stem}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all lifting classes in the module
                new_liftings = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and issubclass(obj, Graph2HypergraphLifting)
                        and obj != Graph2HypergraphLifting
                    )
                }
                liftings.update(new_liftings)
        return liftings


# Create the exports manager
manager = ModuleExportsManager()

# Automatically discover and populate GRAPH2HYPERGRAPH_LIFTINGS
GRAPH2HYPERGRAPH_LIFTINGS = manager.discover_liftings(__file__)

# Automatically generate __all__
__all__ = [
    *GRAPH2HYPERGRAPH_LIFTINGS.keys(),
    "Graph2HypergraphLifting",
    "GRAPH2HYPERGRAPH_LIFTINGS",
]

# For backwards compatibility, create individual imports
locals().update(**GRAPH2HYPERGRAPH_LIFTINGS)
