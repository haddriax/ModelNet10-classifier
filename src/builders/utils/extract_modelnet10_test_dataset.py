import urllib.request
import zipfile
import os
import shutil

EXTRACT_PATH = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\data"
OUTPUT_PATH = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\data\ModelNet10\models_test"

print("Copie des dossiers test...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

base = os.path.join(EXTRACT_PATH, "ModelNet10\models_full")
for classe in os.listdir(base):
    test_src = os.path.join(base, classe, "test")
    test_dst = os.path.join(OUTPUT_PATH, classe, "test")
    if os.path.isdir(test_src):
        shutil.copytree(test_src, test_dst)
        count = len(os.listdir(test_src))
        print(f"  {classe} : {count} objets")


print(f"\nTerminé → {OUTPUT_PATH}/")