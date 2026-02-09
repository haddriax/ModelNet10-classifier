import numpy as np

class OffMeshParser:

    @staticmethod
    def parse_off(lines: list[str], has_header: bool = True)-> tuple[np.ndarray, np.ndarray]:
        """
        Parse an OFF file into an array of vertices and faces.

        Args:
            lines: Lines from the OFF file
            has_header: Whether the file has 'OFF' header line - should almost always be true
            delimiter: Delimiter used in the file (default: space)

        Returns:
            Tuple of (vertices: np.ndarray, faces: np.ndarray, name: str)
        """
        delimiter = ' '
        # First line, usually "OFF"
        line_idx = 0
        if has_header:
            if lines[line_idx].strip() != 'OFF':
                raise ValueError("Invalid OFF file: Missing header")
            line_idx += 1

        # Second line, contains information on the number of faces and vertices
        header: list[str] = lines[line_idx].strip().split(sep=delimiter)
        num_vertices = int(header[0])
        num_faces = int(header[1])
        line_idx += 1

        # Core of the file, iterating on each vertice coordinates
        vertices: np.ndarray = np.empty((num_vertices, 3))
        for i in range(num_vertices):
            vertice_coords = lines[line_idx].strip().split(sep=delimiter)[:3]
            vertices[i] = [float(coord) for coord in vertice_coords]
            line_idx += 1

        # Assume that the faces are triangles (that's the normal way of representing faces)
        faces: np.ndarray = np.empty((num_faces, 3), dtype=int)
        for i in range(num_faces):
            face_data = lines[line_idx].strip().split(sep=delimiter)
            num_vertices_in_face = int(face_data[0])
            face = face_data[1:num_vertices_in_face + 1]
            faces[i] = [int(x) for x in face]
            line_idx += 1

        return vertices, faces