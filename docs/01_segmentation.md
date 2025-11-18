# 01_segmentation.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `load_path_config()` | Lädt `config/path.json` einmalig. | Prüft das Vorhandensein der Pfaddatei, lädt sie mit UTF-8-Encoding und cached das Ergebnis per `lru_cache`, damit alle Aufrufe konsistente Pfade sehen. |
| `load_segmentation_config()` | Lädt die Segmentierungsparameter aus `segmentation.json`. | Öffnet die Konfigurationsdatei, validiert ihre Existenz und stellt das geparste JSON dank Cache global bereit. |
| `read_path_section(name)` | Liest einen benannten Abschnitt aus der Pfadkonfiguration. | Gibt die Datenstruktur zum gewünschten Schlüssel zurück (z. B. `paths`) oder `{}` bei unbekannten Einträgen. |
| `norm_path(value)` | Normalisiert Pfadangaben. | Wandelt konfigurierte Pfade via `os.path.normpath` in das aktuelle Betriebssystemformat und liefert leere Strings für fehlende Werte. |
| `cast_value(cfg, key, default, fn)` | Typisiert konfigurierte Werte. | Holt einen Schlüssel aus einem Dictionary, ersetzt ihn ggf. durch einen Default und führt eine Transformationsfunktion (z. B. `np.array`, `tuple`) aus, um direkt nutzbare Typen zu erzeugen. |
| `load_paths()` | Erzeugt das Pfad-Mapping. | Liest den Abschnitt `paths` und normalisiert jeden Eintrag, sodass globale Variablen wie `raw_dir` und `proc_dir` sofort verwendet werden können. |
| `load_preproc()` | Aggregiert Preprocessing-Parameter. | Fasst HSV-Schwellwerte, Mindestkonturflächen, Warpfeldgrößen und Zielauflösungen aus der Konfiguration zu einem Dictionary für die Segmentierung zusammen. |
| `show_progress(prefix, current, total, bar_len)` | Visualisiert Fortschritt. | Zeichnet einen Textfortschrittsbalken mit `#`/`-`, der während der Batchverarbeitung laufend überschrieben wird. |
| `clear_folder(folder)` | Entfernt das Zielverzeichnis sicher. | Löscht vorhandene Ausgabeordner rekursiv und hebt ggf. Schreibschutz auf, damit jedes Segmentierungslauf mit einer leeren Struktur startet. |
| `scan_images(source_dir)` | Findet alle relevanten Bilddateien. | Durchläuft rekursiv das Quellverzeichnis und yieldet jedes JPG/JPEG/PNG als `(root, name)`-Tupel für die spätere Verarbeitung. |
| `warp_segments(image, warp_cfg)` | Schneidet und normiert Snack-Segmente. | Maskiert das Bild über invertierte HSV-Schwellen, findet Konturen über Mindestfläche, richtet deren Rotationsrechtecke aus, transformiert sie perspektivisch auf das gewünschte Frame und skaliert die Resultate auf die Zielgröße. |
| `segment_folder(source_dir, target_dir, preproc_cfg)` | Segmentiert ein komplettes Eingabeverzeichnis. | Leert das Ziel, lädt alle Bilder, ruft `warp_segments` für jedes Bild auf, legt Klassen-Unterordner an und speichert jede gefundene Kachel; Fortschritt und leere Eingaben werden sauber behandelt. |
| `segment_cli()` | CLI-Einstiegspunkt für die Segmentierung. | Prüft das Vorhandensein des Rohbild-Ordners, legt das Ausgabeziel inkl. Eltern an und startet `segment_folder` mit den global geladenen Konfigurationswerten. |

## Konfiguration: segmentation.json

`config/segmentation.json` stellt alle Parameter für `load_preproc()` bereit.

| Schlüssel | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `preprocessing.preprocess_hsv_lower` | Untere HSV-Schwelle für die Hintergrundmaske. | `[H, S, V]`-Tripel (Standard: `[35, 40, 30]`). Bestimmt, welche Farbtöne als Hintergrund markiert und invertiert werden. |
| `preprocessing.preprocess_hsv_upper` | Obere HSV-Schwelle für die Hintergrundmaske. | Ergänzt die untere Grenze (Standard: `[85, 255, 255]`) und schließt das Maskenintervall. |
| `preprocessing.minimum_contour_area` | Mindestfläche für gültige Konturen. | Ignoriert Konturen unterhalb dieser Pixelzahl (`30000`), um kleine Artefakte auszuschließen. |
| `preprocessing.warp_frame_size` | Breite/Höhe des Zwischen-Warp-Rahmens. | Zwei Werte `[width, height]` (Standard: `[600, 400]`), auf die die Perspektivtransformation angewendet wird, bevor das Ergebnis auf die Zielgröße skaliert wird. |
| `preprocessing.target_width` | Endbreite der gespeicherten Kachel. | Nach dem Warpen wird jedes Segment auf diese Breite skaliert (`400`). |
| `preprocessing.target_height` | Endhöhe der gespeicherten Kachel. | Zielhöhe (`400`), gemeinsam mit `target_width` die finale Auflösung aller Folgeprozesse. |
