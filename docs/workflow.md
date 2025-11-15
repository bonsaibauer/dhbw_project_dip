# Workflow & Parametrisierung

Dieses Dokument beschreibt, wie `main.py` arbeitet. Die folgenden sechs Schritte bilden den kompletten Workflow – jede Überschrift enthält eine Kurzbeschreibung („einfach erklärt“) und darunter die Details mit Bezug auf die Funktionen und Parameter.

## 1. Einstellungen laden & Ordner vorbereiten

**Kurz erklärt**  
Bevor Bilder angefasst werden, definiert der Code alle Pfade, Zielordner und Parameter (z.B. Farbschwellen, Kernelgrößen). So lässt sich das Verhalten zentral steuern.

**Details**  
- Konstanten wie `RAW_DATA_DIR`, `OUTPUT_DIR`, `LOWER_GREEN`, `DEFECT_SPOT_THRESHOLD` usw. liegen am Anfang von `main.py`.  
- `OUTPUT_DIR` wird zu Beginn des Hauptprogramms angelegt (`os.makedirs(OUTPUT_DIR, exist_ok=True)`), damit alle folgenden Schritte ihre Ergebnisse dorthin schreiben können.  
- Die Parametergruppen teilen sich grob in:
  - **Segmentierung** (`LOWER_GREEN`, `UPPER_GREEN`, `CONTOUR_AREA_MIN`, `WARP_SIZE`, `TARGET_WIDTH`, `TARGET_HEIGHT`)
  - **Geometrie-Analyse** (`EPSILON_FACTOR`, `MIN_HOLE_AREA`, `MIN_WINDOW_AREA`, `MAX_CENTER_HOLE_AREA`)
  - **Farbanalyse** (`EROSION_*`, `BLACKHAT_*`, `NOISE_KERNEL_SIZE`, `DEFECT_SPOT_THRESHOLD`)
  - **Validierung/Priorisierung** (`LABEL_PRIORITIES`, `LABEL_CLASS_MAP`)

## 2. Bildliste erstellen

**Kurz erklärt**  
Alle verfügbaren Bilder werden aufgelistet, damit die Segmentierung weiß, wie viel Arbeit ansteht und der Fortschrittsbalken korrekt ist.

**Details**  
- `collect_image_files(source_dir)` durchläuft `data/Images` rekursiv und sammelt jedes `.jpg/.jpeg/.png`.  
- Die Rückgabe ist eine Liste `(root, name)`; dadurch bleibt die Reihenfolge beim späteren Abarbeiten stabil.  
- `prepare_dataset()` nutzt die Länge dieser Liste, um den Fortschritt (`print_progress("  Segmentierung", idx, total_files)`) auszurechnen.

## 3. Segmentierung (Hintergrund entfernen & Perspektive korrigieren)

**Kurz erklärt**  
Jedes Rohbild wird von grünem Hintergrund befreit, zurechtgeschnitten und auf eine einheitliche Größe gebracht. Dadurch erhält die nachfolgende Sortierung saubere, ausgerichtete Snacks.

**Details**  
- `prepare_dataset()` ruft für jedes Bild `run_preprocessing()` auf.  
- `run_preprocessing()`:
  1. Wandelt das Bild in HSV um und erstellt eine Maske mit `LOWER_GREEN`/`UPPER_GREEN`, damit nur der nicht-grüne Snack übrig bleibt.
  2. Sucht Konturen im maskierten Bild und verwirft alles unter `CONTOUR_AREA_MIN`, um Rauschen auszublenden.
  3. Nutzt `cv2.minAreaRect` + `cv2.getPerspectiveTransform`, um das Objekt auf `WARP_SIZE` zu entzerren.
  4. Skaliert auf `TARGET_WIDTH`×`TARGET_HEIGHT`, bevor das Ergebnis gespeichert wird.  
- Die Bilder landen im parallelen Verzeichnis `output/processed/<ursprungsklasse>/...`. Segmentierungsschritte lassen sich über die oben genannten Parameter feinjustieren.

## 4. Geometrie- und Farbprüfung (Sortierung)

**Kurz erklärt**  
Die segmentierten Bilder werden analysiert: Stimmt die Lochanzahl? Gibt es Flecken? Ist die Symmetrie in Ordnung? Danach werden sie einer Klasse zugeordnet und in `output/sorted` kopiert.

**Details**  
- `sort_dataset_manual_rules()` läuft über `output/processed` und ruft folgende Helfer:
  - `get_contours_hierarchy(image)` erzeugt Graustufen, Thresholding und liefert Konturen plus Hierarchie (wichtig für Lochzählung).
  - `analyze_geometry_features(contours, hierarchy)` berechnet `num_windows`, `has_center_hole`, Fensterflächen usw. – gesteuert durch `EPSILON_FACTOR`, `MIN_HOLE_AREA`, `MIN_WINDOW_AREA`, `MAX_CENTER_HOLE_AREA`.
  - `detect_defects(image, spot_threshold=DEFECT_SPOT_THRESHOLD)` trennt Objekt vom Hintergrund, nutzt Morphological Black-Hat (`BLACKHAT_KERNEL_SIZE`, `BLACKHAT_CONTRAST_THRESHOLD`) und prüft, ob die Fleckenfläche größer als `DEFECT_SPOT_THRESHOLD` ist.
- Entscheidungslogik:
  1. Kein Objekt erkannt → `Rest`.
  2. Lochanzahl ≠ 7 → `Bruch` (zu wenig) oder `Rest` (zu viele Fragmente).
  3. Lochanzahl = 7 → Farbprüfung. Bei Flecken → `Farbfehler`, sonst Symmetrie-Bewertung (`SYMMETRY_SENSITIVITY`) und Einstufung als `Normal`.
- Ausgaben:
  - Mit `VERBOSE_SORT_OUTPUT = True` erscheint pro Bild eine Logzeile `[Klasse] Name | Grund | Ziel`.
  - Alle Dateien werden nach `output/sorted/<Klasse>/` kopiert, bei `Normal` zusätzlich mit Symmetriepräfix (`XYZ_`).

## 5. Übersicht & Reasoning der Sortierung

**Kurz erklärt**  
Nach der Sortierung zeigt der Code eine Tabelle mit Anzahl, Anteilen und häufigstem Grund pro Klasse. So sieht man sofort, wie viele Fälle wohin gelaufen sind und warum.

**Details**  
- `sort_dataset_manual_rules()` sammelt Laufzeitstatistiken (`stats_counter`, `reason_counter`).  
- Am Ende wird `print_table()` mit den Spalten *Klasse*, *Anzahl*, *Anteil %*, *Beschreibung* und *Häufigster Grund* aufgerufen.  
- Die Beschreibung kommt aus `CLASS_DESCRIPTIONS`, sodass direkt erkennbar ist, wofür jede Klasse steht.

## 6. Validierung & Fehlersammlung

**Kurz erklärt**  
Die Sortierergebnisse werden mit `data/image_anno.csv` verglichen. Ein Bericht zeigt Genauigkeiten und Fehler, daneben werden falsch eingeordnete Bilder in einen Kontrollordner kopiert.

**Details**  
- `load_annotations()`:
  - Verwendet `normalize_relative_path()` zur Vereinheitlichung der Dateipfade.
  - Zerlegt Mehrfachlabels (`resolve_priority_label()`) und mappt sie über `LABEL_CLASS_MAP` auf die vier Zielklassen.
- `validate_predictions(predictions, annotations, FALSCH_DIR)`:
  1. Zählt Treffer und Fehler pro Klasse.
  2. Gibt zwei Tabellen aus – **Gesamtstatistik** (Bewertet, Treffer, Genauigkeit, Falsch zugeordnet, ggf. „Ohne passende Annotation“) und **Klassenübersicht** (Erwartet, Treffer, Genauigkeit %).
  3. Zeigt die aktuelle Priorisierungskette (z.B. `Bruch > Farbfehler > Rest > Normal`), damit klar ist, warum Mehrfachlabels so aufgelöst wurden.
  4. Kopiert jede Fehlzuordnung nach `output/Falsch` und ergänzt den Dateinamen mit `gt-<korrekt>` und `pred-<erkannt>`.

## Hauptprogramm in Kurzform

1. **Einstellungen & Ordner** – Parameter laden, `output/` anlegen.
2. **Bildliste** – Dateien sammeln (`collect_image_files`).
3. **Segmentierung** – Hintergründe entfernen, Bilder normieren (`prepare_dataset` → `run_preprocessing`).
4. **Sortierung** – Geometrie + Farbe prüfen, Dateien klassifizieren (`sort_dataset_manual_rules` + Helfer).
5. **Übersicht** – Tabellarische Zusammenfassung der Sortierergebnisse (`print_table` in Schritt 5).
6. **Validierung** – Abgleich mit `image_anno.csv`, Report + Kopie der Fehlbilder (`validate_predictions`).

Jeder Schritt baut auf dem vorherigen auf, wodurch der gesamte Prozess von Rohdaten bis Qualitätsreport vollständig automatisiert ist.
