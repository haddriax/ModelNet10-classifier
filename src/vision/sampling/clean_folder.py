import os
import shutil

base_path = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\ModelsDatasetOutput"

for folder in os.listdir(base_path):
    if folder.endswith("(Clone)"):
        full_path = os.path.join(base_path, folder)
        shutil.rmtree(full_path)
        print(f"Supprimé : {folder}")

print("Done")