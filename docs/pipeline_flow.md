# Verarbeitungsschritte der Fryum-Pipeline

Diese Anleitung beschreibt den kompletten Ablauf von der Rohdatei bis zum klassifizierten Ergebnis. Jeder Schritt nennt die verantwortlichen Module, Eingaben/Ausgaben und relevante Artefakte auf der Festplatte.

## 1. Daten einlesen (`app/steps/dataset_loader.py`)

1. `image_anno.csv` wird eingelesen.
2. Für jede Zeile wird entschieden, ob das Bild im Ordner `data/Images/Normal` oder `data/Images/Anomaly` liegt.
3. Die Original-Label werden auf die vier Zielklassen gemappt (`config.AGGREGATED_CLASS_MAPPING`).
4. Ergebnis ist eine Liste von `ImageRecord`-Objekten mit Pfaden, Original- und Zielklassen.

## 2. Segmentierung vorbereiten (`app/pipeline.py`)

1. Der `BackgroundSegmenter` wird mit den gewählten Parametern initialisiert.
2. Für jeden Datensatz wird das entsprechende Bild mit OpenCV geladen.

## 3. Hintergrundentfernung (`app/steps/background_segmentation.py`)

1. **Gaussian Blur** glättet das Bild.
2. **LAB-Umwandlung** extrahiert die L-Komponente.
3. **Otsu-Schwelle** erzeugt eine binäre Maske.
4. Die Maske wird ggf. invertiert (abhängig vom Durchschnittswert).
5. **Median-Filter** reduziert Rauschen.
6. Morphologische Operationen (Close/Open oder Open/Close, ggf. mehrfach) säubern die Maske.
7. Optional bleibt nur die größte Kontur erhalten, anschließend erfolgt ein Bounding-Box-Crop.
8. Ergebnis: `SegmentationResult` mit Maske, Crops, Bounding Box, geblurrtem Bild, etc.

## 4. Feature-Berechnung (`app/steps/feature_extraction.py`)

1. Auf Basis der Maske werden Konturen analysiert (Fläche, Perimeter, Konvexhülle).
2. Shape-Metriken: `area_ratio`, `bbox_ratio`, `elongation`, `solidity`, `roughness`.
3. Symmetrie (`_symmetry_score`) vergleicht linke/rechte Maskenseite.
4. Bild wird nach LAB konvertiert, Maskenpixel liefern Mittelwerte/Standardabweichungen.
5. Farbanteile (`dark_fraction`, `bright_fraction`, `yellow_fraction`, `red_fraction`) basieren auf den UI-Schwellen.
6. Textur: Laplace-Filter mit wählbarem Kernel, Standardabweichung innerhalb der Maske.
7. Alle Features + Metadaten landen in einer Liste, später DataFrame.

## 5. Merkmaltabelle & Klassifikation (`app/pipeline.py`, `app/steps/classifier.py`)

1. Der DataFrame wird gespeichert (`output/reports/feature_table.csv`) und nach `record_id` indiziert.
2. Klassifikator (`RuleBasedClassifier`) traversiert den eingefrorenen Entscheidungsbaum mit den Feature-Spalten.
3. Vorhersagen werden gesammelt und für spätere Auswertung aufbewahrt.

## 6. Evaluierung (`app/steps/evaluator.py`)

1. `evaluate` berechnet Accuracy, Classification Report und Confusion Matrix.
2. Ergebnisse werden in `output/reports/` als CSV/TXT geschrieben.
3. Die Pipeline loggt eine kurze Zusammenfassung.

## 7. Speichern der Ergebnisse (`app/steps/result_writer.py`, `_save_inspection_assets`)

1. Zu jeder Vorhersage werden Inspektionsartefakte (Original, Masken, Overlays, Crops) im Ordner `output/inspection/<Klasse>/<Dateiname>/` abgelegt.
2. Der `ResultWriter` speichert den segmentierten Ausschnitt in `output/classified/<Klasse>/`.
3. Falsch klassifizierte Crops werden zusätzlich unter `output/classified/Falsch/` mit GT/Pred-Suffix abgelegt.
4. Prozess-Logs (Symmetrie, Feature-Zusammenfassung, Schritte) werden pro `ProcessRecord` gespeichert.

## 8. Viewer-Integration (`app/ui/main.py`, `app/ui/viewer.py`)

1. Nach Abschluss erhält der Viewer eine Liste von `ProcessRecord`-Objekten.
2. Tabs und Detailansicht zeigen Masken, Metriken, Prozessschritte und ermöglichen die erneute Ausführung mit geänderten Parametern.

## 9. CLI-/Batch-Ausführung (`run_pipeline.py`, `setup_and_run.bat`)

1. `run_pipeline.py` führt nur die Pipeline aus und öffnet danach den Viewer mit den Ergebnissen.
2. `setup_and_run.bat` richtet die Python-Umgebung ein, startet das UI und streamt Logs live.
