import os
from reconstruct import reconstruct_folder

INPUT_DIR  = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\ModelsDatasetOutput"

OUTPUT_DIR = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\DatasetTypeModelNetCanny"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"ERREUR : dossier introuvable {INPUT_DIR}")
        return

    folders = [
        os.path.join(INPUT_DIR, d)
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]

    if not folders:
        print("ERREUR : aucun sous-dossier trouvé dans INPUT_DIR.")
        return

    print()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success, failed = 0, []

    for folder in folders:
        name = os.path.basename(folder)
        if not os.path.exists(os.path.join(folder, "cameras.json")):
            print(f"[{name}] IGNORÉ : pas de cameras.json")
            failed.append(name)
            continue
        images = [f for f in os.listdir(folder) if f.endswith(".png")]
        if len(images) == 0:
            print(f"[{name}] IGNORÉ : aucune image PNG")
            failed.append(name)
            continue

        try:
            reconstruct_folder(folder, OUTPUT_DIR)
            success += 1
        except Exception as e:
            print(f"[{name}] ERREUR : {e}")
            failed.append(name)

    if failed:
        print("Échecs :", ", ".join(failed))
        

if __name__ == "__main__":
    main()