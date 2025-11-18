# main.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `run_stage(script_name, entry_point)` | Lädt ein Stufenskript dynamisch und ruft dessen CLI-Einstieg. | Führt das angegebene Skript aus dem `scripts/`-Ordner via `runpy.run_path` aus, sucht nach der gewünschten Funktion, verifiziert deren Aufrufbarkeit und startet anschließend die Stage; fehlt die Funktion, wird ein RuntimeError ausgelöst. |
| `run_pipeline()` | Orchestriert alle Pipeline-Stufen in fester Reihenfolge. | Iteriert über `STAGE_LIST`, die Dateinamen und Entry-Points enthalten, und ruft für jeden Eintrag `run_stage` auf, sodass Segmentierung, Bildverarbeitung, Klassifikation, Sortierung und Validierung nacheinander ausgeführt werden. |

## Konfiguration: path.json

`config/path.json` liefert die zentralen Ablageorte für alle Stufen. Beim Laden werden die Pfade normalisiert (`os.path.normpath`).

| Schlüssel | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `paths.raw_image_directory` | Quelle der Rohbilder. | Wurzelverzeichnis der unbearbeiteten Snacks (Standard: `data/Images`). Die Segmentierung liest hier rekursiv alle JPG/JPEG/PNG-Dateien ein und nutzt Unterordner als Klassenbezeichner. |
| `paths.processed_image_directory` | Zielordner der Segmentierung. | `01_segmentation.py` erstellt diesen Ordner neu und legt pro Quellklasse die zugeschnittenen Kacheln ab; spätere Stufen greifen hierauf zu. |
| `paths.sorted_output_directory` | Ausgabe der Sortierphase. | `04_sorting.py` kopiert alle klassifizierten Bilder hierher, nach Klassen gruppiert. Der Ordner wird vor jedem Lauf geleert. |
| `paths.failed_validation_directory` | Sammelstelle für Fehlklassifikationen. | `05_validation.py` legt diesen Unterordner innerhalb des Sortieroutputs an und kopiert dort jede Abweichung (Dateinamen enthalten Ground Truth und Prediction). |
| `paths.annotation_file_path` | CSV mit Ground-Truth-Labels. | Referenz auf `data/image_anno.csv`. Die Validierung nutzt diesen Pfad zum Abgleich zwischen Predictions und manuellen Labels; fehlt die Datei, wird sie übersprungen. |
| `paths.pipeline_csv_path` | Austauschformat zwischen Stufen. | `02_image_processing.py` erzeugt hier die Feature-CSV. `03_classification.py` ergänzt Zielklassen/Begründungen, `04_sorting.py` und `05_validation.py` lesen sie erneut für Kopierziele bzw. Reports. |
