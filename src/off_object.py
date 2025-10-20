import os
import re
from pathlib import Path

class OffObject:

    regex_expr = re.compile( r'_+\d+$')
    delimiter = ' '

    def __init__(self, vertices: list[list[float]], faces: list[list[int]], name: str, path: Path = None):
        self.path = None
        self.vertices = vertices
        self.faces = faces
        self.name = name

    def __str__(self):
        rep = f'OffObject: {self.name}, {len(self.vertices)} vertices, {len(self.faces)} faces\n'
        return rep

    @staticmethod
    def from_lines_list(lines: list[str], name: str, has_header: bool = True) -> 'OffObject':
        line_idx = 0
        if has_header:
            if lines[line_idx].strip() != 'OFF':
                raise ValueError("Invalid OFF file: Missing header")
            line_idx += 1

        header: list[str] = lines[line_idx].strip().split(sep=OffObject.delimiter)
        num_vertices = int(header[0])
        num_faces = int(header[1])
        line_idx += 1

        vertices: list[list[float]] = []
        for i in range(num_vertices):
            vertex: list[float] = [float(coord) for coord in
                                   lines[line_idx].strip().split(sep=OffObject.delimiter)[:3]
                                   ]
            vertices.append(vertex)
            line_idx += 1

        faces: list[list[int]] = []
        for i in range(num_faces):
            face_data = [int(x) for x in lines[line_idx].strip().split(sep=OffObject.delimiter)]
            face: list[int] = face_data[1:face_data[0]+1]
            faces.append(face)
            line_idx += 1

        return OffObject(vertices, faces, name=name)

    @staticmethod
    def load_from_file(file_path: Path) -> 'OffObject':
        name = re.sub(OffObject.regex_expr, '', file_path.stem)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OFF file not found: {file_path}")

        lines: list[str]
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid OFF file format in {file_path}: {e}")

        obj = OffObject.from_lines_list(lines, name=name, has_header=True)
        obj.path = file_path
        return obj


if __name__ == "__main__":
    test_path = Path("night_stand_0001.off")
    oof_object = OffObject.load_from_file(test_path)
    print(oof_object)