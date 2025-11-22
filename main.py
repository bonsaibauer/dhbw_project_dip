import os
import sys

from scripts import segmentierung
from scripts import bruch
from scripts import rest
from scripts import farb
from scripts import symmetrie
from scripts import ergebnis

if __name__ == '__main__':
    RAW_DATA_DIR = os.path.join("data", "Images")
    OUTPUT_DIR = os.path.join("output")
    PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed")
    SORTED_DATA_DIR = os.path.join(OUTPUT_DIR, "sorted")
    ANNO_FILE = os.path.join("data", "image_anno.csv")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SORTED_DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Fehler: Quellordner '{RAW_DATA_DIR}' nicht gefunden.")
        user_input = input("Bitte Pfad zum Quellordner (Ordnername/Images) eingeben: ").strip()

        if not user_input:
            print("Kein Pfad angegeben. Programm wird beendet.")
            sys.exit(1)

        RAW_DATA_DIR = user_input

        if not os.path.isdir(RAW_DATA_DIR):
            print(f"Fehler: Angegebener Quellordner '{RAW_DATA_DIR}' nicht gefunden.")
            sys.exit(1)

    segmentierung.prepare_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print("Fehler: Keine Bilder verarbeitet.")
        sys.exit(1)

    bruch.sort_images(PROCESSED_DATA_DIR, SORTED_DATA_DIR)
    rest.run_complexity_check(SORTED_DATA_DIR)
    farb.run_color_check(SORTED_DATA_DIR)
    symmetrie.run_symmetry_check(SORTED_DATA_DIR)

    if os.path.exists(ANNO_FILE):
        ergebnis.evaluate_results(SORTED_DATA_DIR, ANNO_FILE)

    print("\nPipeline abgeschlossen.")
