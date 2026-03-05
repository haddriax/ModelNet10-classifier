import os
from reconstruct_stereovision import reconstruct_stereo

INPUT_DIR  = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\ScreenShots"

OUTPUT_DIR = r"C:\Users\fanny\OneDrive\Bureau\Cours_CS\DEEPL\ModelNet10-classifier\unity\Visual_V0\Assets\DatasetTypeModelNetStereovision"

def main():
    """Lance la reconstruction stéréo sur tous les objets."""
    if not os.path.exists(INPUT_DIR):
        print(f"ERREUR : dossier introuvable → {INPUT_DIR}")
        return

    folders = [
        os.path.join(INPUT_DIR, d)
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]

    print(f"{len(folders)} objet(s) trouvé(s)")
    success, failed = 0, []

    for folder in folders:
        name = os.path.basename(folder)
        try:
            reconstruct_stereo(folder, OUTPUT_DIR)
            success += 1
        except Exception as e:
            print(f"  [{name}] ERREUR : {e}")
            failed.append(name)

    print(f"\n=== Batch terminé : {success} réussi(s), {len(failed)} échoué(s) ===")
    if failed:
        print("Échecs :", ", ".join(failed))

if __name__ == "__main__":
    main()


