# Workflow: Pipeline & Entscheidungslogik

Dieses Dokument beschreibt die aktuelle Snack-Pipeline, wie sie durch `main.py` orchestriert wird. Jeder Abschnitt beleuchtet:

- **Welche Skripte und Funktionen** beteiligt sind,
- **Welche Artefakte** an die nächste Stufe weitergereicht werden und
- **Welche Konfigurationen** die Verarbeitung steuern.

Alle Angaben beziehen sich auf den Repository-Stand vom November 2025.

---

## 1. Gesamtüberblick (`main.py`)

- `STAGE_LIST` definiert die Abfolge der Skripte:  
  1. `01_segmentation.segment_cli`  
  2. `02_image_processing.process_cli`  
  3. `03_classification.classify_cli`  
  4. `04_sorting.sort_cli`  
  5. `05_validation.validate_cli`
- `run_pipeline()` iteriert diese Liste und ruft je Eintrag `run_stage(script_name, entry_point)` auf. `run_stage` lädt das Skript mittels `runpy.run_path`, sucht den benannten Einstiegspunkt und führt ihn aus. Dadurch ist jede Stage isoliert testbar und erhält ihre Konfiguration aus den JSON-Dateien im `config/`-Ordner.

---

## 2. Konfiguration & Pfadverwaltung

| Datei | Zweck |
| --- | --- |
| `config/path.json` | Pfade für Rohdaten, Segmentierungsoutput, Pipeline-CSV, Sortierordner, Validierungsordner, Annotationen. |
| `config/segmentation.json` | HSV-Schwellwerte, Mindestkonturfläche, Warp-Größe und Zielauflösung für `01_segmentation.py`. |
| `config/image_processing.json` | Geometrie- und Spot/Farb-Parameter (Kernel, Iterationen, Toleranzen) für `02_image_processing.py`. |
| `config/classification.json` | Regeldefinitionen, Label-Prioritäten und Mapping Label → Klasse für `03_classification.py` sowie Validierung. |

Alle Loader (`load_path_config`, `load_segmentation_config`, `load_image_config`, `class_config`) verwenden `lru_cache`, sodass jede Datei nur einmal eingelesen wird. Pfade werden konsequent mit `os.path.normpath` normalisiert.

---

## 3. Stufe 1 – Segmentierung (`scripts/01_segmentation.py`)

1. **Setup**  
   - `load_paths()` extrahiert z. B. `raw_image_directory` und `processed_image_directory`.  
   - `load_preproc()` erzeugt ein Dictionary mit HSV-Schwellen (`preprocess_hsv_lower/upper`), Mindestkonturfläche, Warp-Größe und Zielauflösung.
2. **Verarbeitung**  
   - `segment_folder(source_dir, target_dir, preproc_cfg)` leert das Ziel via `clear_folder`, scannt alle Bilder (`scan_images`) und ruft für jede Datei `warp_segments(image, preproc_cfg)` auf.  
   - `warp_segments` maskiert den Snack (HSV-Inversionsmaske), filtert Konturen nach Mindestfläche, richtet sie über `cv2.getPerspectiveTransform` aus und skaliert sie auf `target_width/height`. Jedes Ergebnis wird im Zielordner unterhalb des Klassen-Unterordners gespeichert.
3. **Feedback**  
   - `show_progress("Segmentierung", idx, total)` zeigt während des Batch-Laufs einen Textbalken an.

**Output:** Ein bereinigtes Set segmentierter Snacks im `processed_image_directory`.

---

## 4. Stufe 2 – Bildverarbeitung & Feature-Export (`scripts/02_image_processing.py`)

1. **Pfad- und Parameter-Setup**  
   - `load_paths()` liefert `processed_image_directory` (Input) und `pipeline_csv_path` (Output).  
   - `load_geometry()` sammelt Geometrie-Parameter (z. B. `polygon_epsilon_factor`, `minimum_window_area`).  
   - `load_spot()` liefert sämtliche Spot-/Farb-Einstellungen (Kernelgrößen, Iterationen, Schwellwerte, Prozentile).
2. **Feature-Ermittlung (`analyze_image`)**  
   - Liest das Bild, normalisiert Pfade (`normalize_path`), erkennt Anomalien über den Ordnernamen.  
   - `extract_contours` liefert Konturen + Hierarchie → `geometry_stats` berechnet Hauptkonturfläche, Konvexität, Fensteranzahl/-flächen, Mittellochstatus, Fragmentanzahl, Außenkonturen sowie Außenbruch- und Fensterabweichungsmetriken (`outer_break_metrics`, `window_deviation_metrics`).  
   - `symmetry_score` bestimmt eine 6-fach-Rotationssymmetrie (0–100 %).  
   - Für Anomalien wird `detect_spots` ausgeführt: Maskenaufbau, Blackhat, Kontrastschwelle, Rauschfilter, innere/feine Spotprüfung sowie Statistiken (`color_texture_stddev`, `color_lab_stddev`, `color_dark_delta`, …). Normale Bilder erhalten Nullwerte.  
   - Alle Werte werden als Strings formatiert bzw. für Listen (`geometry_window_area_list`) JSON-kodiert.
3. **CSV-Erstellung (`process_folder`)**  
   - Durchläuft rekursiv alle Bilder, ruft `analyze_image` auf, zeigt Fortschritt und schreibt das Ergebnis in `pipeline_csv_path`. Bei leeren Inputs wird nur der Header `CSV_FIELDS` erzeugt.

**Output:** Eine CSV mit allen Geometrie-, Symmetrie- und Farbmerkmalen pro Snack.

---

## 5. Stufe 3 – Klassifikation (`scripts/03_classification.py`)

1. **Konfiguration**  
   - `map_labels()` (Label → Klassenname) und `rank_labels()` (Priorität) aus `classification.json`.  
   - `rule_list()` liefert sämtliche Regelobjekte (Basis-Score, Bedingungen, Gewichte, Mindestscore).
2. **Feature-Aufbereitung (`extract_metrics`)**  
   - Parsed Fensterflächen/-anzahl, Mittellochflag, berechnet `window_size_variance_score`, `geometry_hull_ratio`, Kantenkennzahlen, Fragment- und Außenkonturen, Außenradienmetriken sowie Spot-/Farbwerte (inkl. `color_issue_detected`).  
3. **Regel-Engine**  
   - `score_rule()` prüft jede Bedingung via `match_metric` (Operatoren: `>=`, `<=`, `>`, `<`, `==`, `!=`, `between`, `in`), addiert Gewichte und protokolliert Gründe.  
   - `eval_rules()` sammelt alle erfüllten Regeln.  
   - `bruch_decisions()` ergänzt heuristische Regeln speziell für Anomalien: Fensterabweichungen (`INNER_BREAK_*`) und Außenradien (`OUTER_BREAK_*`) werden genutzt, um zusätzliche Entscheidungen wie „corner or edge breakage“, „middle breakage“ oder „bruch“ einzutragen.  
   - `pick_decision()` sortiert nach Score und Priorität; greift keine Regel, liefert `fallback_pick()` „rest“ (Anomalie) oder „normal“.
4. **CSV-Update (`classify_csv`)**  
   - Schreibt `target_label`, `target_class`, `reason` und `geometry_window_size_variance_score` zurück in `pipeline_csv_path`.  
   - Gibt zusätzlich eine Liste von Predictions (`relative_path`, `predicted`, `label`, `reason`) zurück, die Sortierung und Validierung verwenden.  
   - Fortschritt wird mit `show_progress("Klassifizierung", ...)` angezeigt.

**Output:** Die Pipeline-CSV enthält pro Snack finale Klasse, Label und Begründung.

---

## 6. Stufe 4 – Sortierung & Statistik (`scripts/04_sorting.py`)

1. **Vorbereitung**  
   - `sort_images(csv_path, sorted_dir, log_progress)` liest die klassifizierte CSV, leert den Sortierordner (`clear_folder`) und legt Standardklassenordner an (weitere Klassen werden dynamisch erzeugt).
2. **Dateikopie pro Datensatz**  
   - `resolve_destination_name` bestimmt den Zielnamen (für Klasse „Normal“ optional mit `symmetry_score`-Präfix via `prefixed_name`).  
   - Jedes Bild wird aus `row["source_path"]` in den passenden Klassenordner kopiert; `sorted_path` und `destination_filename` werden in der CSV aktualisiert.  
   - Fortschritt wird über `show_progress("Sortierung", ...)` angezeigt.
3. **Reporting**  
   - Nach Abschluss werden Anzahl und Anteil pro Klasse gezählt und als Tabelle über `render_table` ausgegeben.

**Output:** Strukturierter Sortierordner sowie CSV-Spalten `sorted_path` und `destination_filename`.

---

## 7. Stufe 5 – Validierung (`scripts/05_validation.py`)

1. **Daten laden**  
   - `load_preds()` liest nur Zeilen mit `target_class` aus der CSV.  
   - `load_annos()` lädt `annotation_file_path`, normalisiert alle Pfade (`normalize_path`) und wählt bei mehrfachen Labeln das priorisierte Label (`select_label`) bevor es per `map_labels` auf eine Klasse gemappt wird.
2. **Vergleich & Statistik (`check_preds`)**  
   - Prüft, ob Annotationen vorhanden sind; fehlen sie, wird die Validierung übersprungen.  
   - Löscht ggf. `failed_validation_directory`, iteriert über alle Vorhersagen, vergleicht sie mit den Annotationen und sammelt Treffer/Mismatches pro Klasse (`per_class`).  
   - Gibt eine Tabelle mit Gesamtstatistik („Gesamt“, „Ohne passende Annotation“, Klassenzeilen) aus; Genauigkeiten werden in Prozent ausgegeben.  
   - `build_chain()` liefert zusätzlich den Prioritätsstring („Bruch > Farbfehler > …“).  
   - `copy_miss()` kopiert jede Fehlklassifikation ins Fehlerverzeichnis; der Dateiname enthält Ground Truth und Prediction (`..._gt-Bruch_pred-Rest.jpg`).

**Output:** Konsolenreport + Ordner mit allen Fehlklassifikationen.

---

## 8. Hilfsfunktionen & wiederkehrende Konzepte

- **Fortschrittsbalken** (`show_progress`): Jede Stage nutzt denselben Stil (Label, Balken, Prozent, `current/total`).  
- **Pfadnormalisierung**: `normalize_path` in Bildverarbeitung/Validierung sorgt dafür, dass CSV-Paths und Annotationen im selben Format vorliegen (`Data/Images/...`).  
- **Tabellenlayout** (`render_table`): Wird in Sortierung und Validierung verwendet, um Spalten ausgerichtet im Terminal auszugeben.  
- **Fehlerrobustheit**: Alle Loader prüfen auf Dateiexistenz, leere Inputs führen zu informativen Meldungen (z. B. nur CSV-Header, Hinweis „Keine Bilder gefunden“), Ordner werden bei Bedarf neu erstellt (`os.makedirs(..., exist_ok=True)`).

---

## 9. Artefakte & Datenfluss

| Stage | Eingabe | Ausgabe |
| --- | --- | --- |
| Segmentierung | `raw_image_directory` | `processed_image_directory` (Warp-Kacheln) |
| Bildverarbeitung | `processed_image_directory` | `pipeline_csv_path` mit Geometrie-/Farbmerkmalen |
| Klassifikation | `pipeline_csv_path` | Aktualisierte CSV (`target_*`, `reason`), Predictions-Liste |
| Sortierung | Aktualisierte CSV, Originalbilder | `sorted_output_directory`, CSV-Felder `sorted_path`, `destination_filename` |
| Validierung | Aktualisierte CSV, `annotation_file_path` | Terminalstatistik, `failed_validation_directory` |

Die Pipeline kann vollständig über `python main.py` oder stageweise durch Aufruf der jeweiligen CLI-Funktion betrieben werden. Jede Stufe erwartet das Artefakt ihres Vorgängers und lässt sich daher auch einzeln testen bzw. neu starten.
