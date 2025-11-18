# 02_image_processing.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `load_json(path, error_msg)` | Generischer JSON-Loader mit Fehlermeldung. | Prüft, ob die Datei existiert, lädt sie UTF-8-kodiert und gibt das geparste Objekt zurück; fehlt sie, wird eine `FileNotFoundError` mit Kontext ausgelöst. |
| `load_path_config()` | Cached Loader für `config/path.json`. | Ruft `load_json` für die Pfaddefinitionen auf, speichert das Ergebnis via `lru_cache` und stellt es anderen Funktionen ohne erneutes Dateilesen bereit. |
| `load_image_config()` | Cached Loader für `config/image_processing.json`. | Lädt alle Bildverarbeitungsparameter einmalig und macht sie über weitere Hilfsfunktionen verfügbar. |
| `norm_path(path_value)` | Normalisiert Pfade oder liefert einen Leerstring. | Nutzt `os.path.normpath`, um Konfigurationspfade in einheitliches OS-Format zu bringen, wobei fehlende Werte als `""` zurückgegeben werden. |
| `read_path_section(cfg_name)` | Greift auf Abschnitte aus `path.json` zu. | Gibt das Unterobjekt zum gewünschten Schlüssel (z. B. `paths`) zurück oder `{}` bei unbekannten Abschnitten. |
| `read_image_section(cfg_name)` | Greift auf Abschnitte aus `image_processing.json` zu. | Wird von `load_geometry` und `load_spot` verwendet, um logisch gruppierte Parameterblöcke zu beziehen. |
| `pull_value(cfg, key, default, transform)` | Liest Werte mit optionaler Transformation. | Holt einen Schlüssel, ersetzt fehlende Werte durch Default und führt eine Transformationsfunktion (z. B. `tuple`) aus, um sofort verwendbare Typen zu erhalten. |
| `load_paths()` | Baut das Pfad-Mapping. | Iteriert über den `paths`-Abschnitt, normalisiert alle Werte und liefert ein Dictionary für wichtige Orte wie `processed_image_directory` und `pipeline_csv_path`. |
| `load_geometry()` | Aggregiert Geometrie-Parameter. | Liest Grenzwerte wie Polygon-Epsilon, minimale Fenster-/Lochflächen oder Fragmentflächen und fasst sie in einem Dictionary für spätere Auswertungen zusammen. |
| `load_spot()` | Aggregiert Spot/Farb-Parameter. | Stellt Kernelgrößen, Iterationszahlen, Kontrastschwellen, Mindestflächen sowie Quotengrenzen zur Verfügung, die die Farbprüfung steuern. |
| `normalize_path(path)` | Vereinheitlicht Bildpfade für Annotationen. | Ersetzt Backslashes durch Slashes, entfernt den Präfix `Data/Images/` und liefert relative Pfade ohne führende `/`, damit Annotationen und Pipeline-CSV übereinstimmen. |
| `show_progress(prefix, current, total, bar_len)` | Fortschrittsbalken für Stapeloperationen. | Berechnet den Fortschrittsanteil und rendert eine einzeilige Statusanzeige, solange es mindestens ein Element zu verarbeiten gibt. |
| `bool_text(value)` | Gibt boolesche Werte als String zurück. | Konvertiert Wahrheitswerte in `"true"` bzw. `"false"`, um CSV-Felder konsistent zu halten. |
| `build_masks(image, ero_kernel, ero_iterations)` | Erstellt Objekt- und Analysemaske. | Konvertiert das Bild nach Grau, segmentiert den Snack per Schwellwert, erodiert die Analysemaske mit angegebenem Kernel/Iterationen und liefert Graybild, Objekt- und Analysemasken zurück. |
| `make_blackhat(gray, kernel)` | Hebt dunkle Flecken hervor. | Führt eine Blackhat-Morphologie auf dem Graubild aus, sodass dunkle Defekte gegenüber der Umgebung betont werden. |
| `check_contrast(blackhat_img, contrast_threshold)` | Schwellwertet das Blackhat-Bild. | Erstellt eine Defektmaske, indem Pixel oberhalb des Kontrastschwellwerts auf 255 gesetzt werden; dient als Basis für die spätere Spot-Analyse. |
| `filter_spots(mask_defects, mask_analysis, noise_kernel)` | Filtert und beschränkt Defektmasken. | Begrenzt Defekte auf den Snackbereich und entfernt Kleinstrauschen mittels Morphology-Open mit dem konfigurierten Rauschkernel. |
| `texture_stats(gray, mask_analysis, dark_percentile)` | Berechnet Texturkennzahlen. | Extrahiert Pixel innerhalb der Analysemaske, bestimmt Standardabweichung, Median und das angegebene Dunkel-Prozentil und liefert zusätzlich das Dark-Delta (Median minus Perzentil). |
| `lab_stats(image, mask_analysis)` | Misst Farbstreuung in LAB. | Wandelt das Bild nach LAB, wählt den a-Kanal innerhalb der Maske und gibt die Standardabweichung zurück, um Farbrauschen zu quantifizieren. |
| `erode_mask(mask, kernel, iterations)` | Schrumpft Maskenbereiche sauber ein. | Führt eine optionale Erosion aus und liefert sowohl die resultierende Maske als auch deren Pixelanzahl für weitere Flächenvergleiche. |
| `spot_ratio(spot_area, object_area)` | Stabilisiert Flächenquotienten. | Berechnet den Anteil der Spotfläche an der Objektfläche, wobei Divisionen durch Null via `max(1, object_area)` vermieden werden. |
| `check_primary(spot_area, object_area, inner_spot_area, ratio_limit, inner_ratio_limit, spot_threshold)` | Bewertet die Hauptbedingungen für Farbdefekte. | Prüft Mindestfläche, Verhältnis zur Objektfläche sowie Anteil der inneren Spots und liefert `True`, wenn alle Kriterien erfüllt sind. |
| `refine_spots(mask_obj, mask_defects, noise_kernel, ero_kernel, fine_iterations, inner_iterations, inner_ratio_limit, fine_ratio, spot_final_threshold)` | Führt eine feinere Spotprüfung aus. | Erodiert das Objekt stärker, filtert erneut Defekte, vergleicht resultierende Flächen gegen Ratio- und Innenschwellen und erkennt kleinere Defekte, falls diese trotz Hauptprüfung auftreten. |
| `debug_view(blackhat_img, mask_analysis)` | Liefert optional ein Debugbild. | Schneidet das Blackhat-Ergebnis mit der Analysemaske und gibt das Bild zurück, falls Debug-Ausgabe gewünscht ist. |
| `detect_spots(image, settings, debug=False)` | Orchestriert die Farb-/Texturanalyse. | Erzeugt Masken, berechnet Blackhat + Kontrast, filtert Spots, berechnet Textur- und LAB-Kennzahlen, prüft Haupt- und Feinbedingungen, erstellt ein Ergebnis-Dictionary (plus Debugbild) mit Flag, Flächen- und Statistikwerten. |
| `symmetry_score(image)` | Bestimmt einen Rotationssymmetrie-Score (0–100 %). | Ermittelt den Schwerpunkt der Snack-Maske, dreht sie sechsmal um 60°, verschneidet alle Rotationen und berechnet den Anteil symmetrischer Pixel; Asymmetrien reduzieren den Score. |
| `extract_contours(image)` | Liefert Konturen und Hierarchie. | Schwellenwertet das Graubild und ruft `cv2.findContours` mit `RETR_TREE`, um äußere und innere Konturen inklusive Hierarchiebeziehungen zu erhalten. |
| `outer_break_metrics(contour, smooth_window, ratio_threshold)` | Bewertet die Außenkontur auf Brüche. | Berechnet Radien vom Schwerpunkt aus, glättet optional, bestimmt minimale Ratio zum Median sowie Anteil bzw. Anzahl der Punkte unterhalb des Schwellwerts und liefert Kennzahlen zur äußeren Beschädigung. |
| `window_deviation_metrics(window_areas, top_count, deviation_tolerance)` | Bewertet Fensterflächen auf Ausreißer. | Sortiert Fensterflächen, nimmt die größten `top_count`, berechnet deren Mittelwert, maximale relative Abweichung sowie Anzahl (inkl. fehlender Fenster), die die Toleranz überschreiten. |
| `geometry_stats(contours, hierarchy, geo_cfg)` | Aggregiert geometrische Merkmale. | Identifiziert die Hauptkontur, berechnet Flächen/Segmentzahlen, zählt Fragmente und Fenster (inkl. Mittellochprüfung), entfernt ein eventuell falsch zugeordnetes Mittelloch und ergänzt die Ergebnisse um Außenbruch- sowie Fensterabweichungsmetriken. |
| `analyze_image(img_path, source_root, geo_cfg, spot_cfg)` | Erstellt den Feature-Datensatz pro Bild. | Lädt das Bild, erkennt Anomalien an der Pfadstruktur, ruft Kontur-/Geometrie- und Symmetrieanalyse auf, führt bei Bedarf `detect_spots` aus und formatiert sämtliche Kennzahlen als Strings/JSON für den CSV-Export. |
| `process_folder(source_dir, csv_path, geo_cfg, spot_cfg)` | Stapelverarbeitung für einen Ordner. | Prüft das Eingabeverzeichnis, sammelt Bilddateien, ruft `analyze_image` für jede Datei auf, zeigt Fortschritt, schreibt alle Ergebnisse als CSV (oder nur Header bei leeren Inputs) und legt das Zielverzeichnis bei Bedarf an. |
| `process_cli()` | CLI-Einstieg für die Bildverarbeitung. | Startet `process_folder` mit den aus der Konfiguration geladenen Pfaden und Parameter-Dictionaries und bildet damit die zweite Pipeline-Stufe. |

## Konfiguration: image_processing.json

`config/image_processing.json` liefert Parameterblöcke `geometry` und `spot`.

### Abschnitt `geometry`

| Schlüssel | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `geometry.polygon_epsilon_factor` | Glättung für Konturapproximation. | Multiplikator für den Kontur-Perimeter (`cv2.approxPolyDP`), Standard `0.04`. Niedrigere Werte erhalten mehr Details. |
| `geometry.minimum_hole_area` | Mindestfläche für Innenkonturen. | Löcher unterhalb `100` px werden ignoriert. |
| `geometry.minimum_window_area` | Mindestfläche für Fenster. | Nur Löcher ab `500` px zählen als reguläre Fenster. |
| `geometry.maximum_center_area` | Obergrenze für Mittelloch. | Löcher ≤ `3000` px mit vielen Ecken markieren ein zentrales Loch. |
| `geometry.minimum_fragment_area` | Mindestfläche für Fragmente. | Außenkonturen müssen ≥ `6000` px groß sein, um als Fragment zu gelten. |

### Abschnitt `spot`

| Schlüssel | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `spot.erosion_kernel_size` | Kernel für Maskenerosion. | `[w, h]`-Array (Standard `[5, 5]`) für `np.ones`. |
| `spot.erosion_iterations` | Grund-Erosionsschritte. | Anzahl Iterationen (`4`), bevor die Analysemaske steht. |
| `spot.blackhat_kernel_size` | Kernel für Blackhat-Filter. | `[15, 15]` definiert das Structuring Element. |
| `spot.blackhat_contrast_threshold` | Schwelle für Defektmasken. | Pixel mit Grauwert ≥ `30` gelten als potenzielle Spots. |
| `spot.noise_kernel_size` | Kernel zum Entfernen von Rauschen. | `[2, 2]` für Morphology-Open zur Rauschunterdrückung. |
| `spot.minimum_spot_area` | Mindestfläche für Defekte. | Spotfläche muss ≥ `60` px sein, um zu zählen. |
| `spot.spot_area_ratio` | Mindestanteil zur Objektfläche. | Verhältnis `spot_area/object_area` muss ≥ `0.0008` sein. |
| `spot.fine_erosion_iterations` | Zusatzerosion für Feinprüfung. | Anzahl Iterationen (`1`), falls Hauptprüfung fehlschlägt. |
| `spot.inner_erosion_iterations` | Erosion für inneren Bereich. | `1` zusätzlicher Schritt, um `inner_spot_ratio` zu messen. |
| `spot.inner_spot_ratio` | Mindestanteil innerer Flecken. | `inner_spot_area/spot_area` muss ≥ `0.3` sein. |
| `spot.fine_spot_ratio` | Verhältnislimit in der Feinprüfung. | `fine_spot_area/fein erodiertes Objekt` ≥ `0.4`. |
| `spot.fine_spot_area` | Mindestfläche der Feinprüfung. | Spots aus der Feinprüfung benötigen ≥ `40` px. |
| `spot.dark_percentile` | Perzentil für Dark-Delta. | Prozentwert (`5`), der mit dem Median verglichen wird, um `color_dark_delta` zu bilden. |
