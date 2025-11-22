import os
import sys

from scripts import segmentierung
from scripts import bruch
from scripts import rest
from scripts import farb
from scripts import symmetrie
from scripts import ergebnis


def resolve_all_paths():
    def valid(base):
        imgs = os.path.join(base, "Images")
        return all(
            [
                os.path.isdir(base),
                os.path.isdir(os.path.join(imgs, "Normal")),
                os.path.isdir(os.path.join(imgs, "Anomaly")),
                os.path.isfile(os.path.join(base, "image_anno.csv")),
            ]
        )

    base_dir = "data"
    if not valid(base_dir):
        base_dir = input("Pfad zu 'data' mit Images/Normal, Images/Anomaly und image_anno.csv: ").strip()
        if not base_dir or not valid(base_dir):
            print("Fehler: Gültige Datenstruktur nicht gefunden. Programm wird beendet.")
            sys.exit(1)

    output_dir = "output"
    return {
        "base": base_dir,
        "raw": os.path.join(base_dir, "Images"),
        "anno": os.path.join(base_dir, "image_anno.csv"),
        "output": output_dir,
        "processed": os.path.join(output_dir, "processed"),
        "sorted": os.path.join(output_dir, "sorted"),
    }


if __name__ == '__main__':
    p = resolve_all_paths()

    segmentierung.prepare_dataset(p["raw"], p["processed"])
    if not os.listdir(p["processed"]):
        print("Fehler: Keine Bilder verarbeitet.")
        sys.exit(1)

    bruch.sort_images(p["processed"], p["sorted"])
    rest.run_complexity_check(p["sorted"])
    farb.run_color_check(p["sorted"])
    symmetrie.run_symmetry_check(p["sorted"])
    ergebnis.evaluate_results(p["sorted"], p["anno"])

    print("\nPipeline abgeschlossen.")
