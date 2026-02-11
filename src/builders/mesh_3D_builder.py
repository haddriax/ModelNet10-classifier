import re
from pathlib import Path

from src.builders.utils.format_parser import OffMeshParser
from src.geometry import Mesh3D


class Mesh3DBuilder:
    """Builder for creating Mesh3D objects from files."""

    NAME_REGEX = re.compile(r'_\d+$')
    """Extract class name from ModelNet10 filename format (e.g., 'chair_0001' -> 'chair')"""

    @staticmethod
    def from_off_file(path: Path) -> Mesh3D:
        """
        Load Mesh3D from OFF file.

        Args:
            path: Path to .off file

        Returns:
            Mesh3D object
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Extract class name (remove trailing numbers)
        name = re.sub(Mesh3DBuilder.NAME_REGEX, '', path.stem)

        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Failed to read {path}: {e}")

        try:
            vertices, faces = OffMeshParser.parse_off(lines)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid OFF format in {path}: {e}")

        return Mesh3D(vertices=vertices, faces=faces, name=name)