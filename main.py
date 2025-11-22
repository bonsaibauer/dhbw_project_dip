import os
import sys

# EIGENE MODULE IMPORTIEREN
# Module liegen nun im Unterordner "scripts"
from scripts import segmentierung
from scripts import bruch
from scripts import rest
from scripts import farb
from scripts import symmetrie  # <--- NEU IMPORTIERT
from scripts import ergebnis

# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == '__main__':
    
    # PFADE
    RAW_DATA_DIR = os.path.join("data", "Images")
    OUTPUT_DIR = os.path.join("output")
    PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed")
    SORTED_DATA_DIR = os.path.join(OUTPUT_DIR, "sorted")
    ANNO_FILE = os.path.join("data", "image_anno.csv")

    # Output-Verzeichnisse anlegen, falls nicht vorhanden
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SORTED_DATA_DIR, exist_ok=True)

    # Check
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

    # 1. SEGMENTIERUNG
    segmentierung.prepare_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print("Fehler: Keine Bilder verarbeitet.")
        sys.exit(1)

    # 2. GEOMETRIE (Bruch)
    # Sortiert grob in Normal, Bruch, Rest
    bruch.sort_images(PROCESSED_DATA_DIR, SORTED_DATA_DIR)

    # 3. KOMPLEXITÄT (Rest / Überlagerung)
    # Prüft Normal & Bruch auf "Chaos" (zu viele Kanten) und schiebt sie nach Rest
    rest.run_complexity_check(SORTED_DATA_DIR)

    # 4. OBERFLÄCHE (Farbe)
    # Prüft Normal auf Flecken
    farb.run_color_check(SORTED_DATA_DIR)

    # 4.5 SYMMETRIE-CHECK (Nur für Klasse Normal)
    # Berechnet Score und benennt Dateien um (Sortierung)
    symmetrie.run_symmetry_check(SORTED_DATA_DIR)

    # 5. EVALUIERUNG
    if os.path.exists(ANNO_FILE):
        ergebnis.evaluate_results(SORTED_DATA_DIR, ANNO_FILE)
    
    print("\nPipeline abgeschlossen.")
