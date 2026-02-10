import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import DATA_DIR, PROJECT_ROOT
from pathlib import Path

model_files = list(DATA_DIR.rglob("*.off"))
directory_obj = Path( PROJECT_ROOT / "data" / "ModelNet10" / "models_obj")
directory_obj.mkdir(parents=True, exist_ok=True)


def off_to_obj(off_file, obj_file):
    with open(off_file, 'r') as f:
        lines = f.readlines()
    
    if lines[0].strip() != 'OFF':
        return
    
    counts = lines[1].split()
    n_vertices = int(counts[0])
    
    with open(obj_file, 'w') as out:
        for i in range(2, 2 + n_vertices):
            vertex = lines[i].split()
            out.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        for i in range(2 + n_vertices, len(lines)):
            face = lines[i].split()
            if len(face) > 1:
                indices = [str(int(x) + 1) for x in face[1:]]
                out.write(f"f {' '.join(indices)}\n")

for off_file in model_files:
    obj_file = directory_obj / (off_file.stem + ".obj")
    off_to_obj(off_file, obj_file)