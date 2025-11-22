# rest.py

## remove_small_artifacts(binary_img, min_area)
- Kurzbeschreibung: Entfernt kleine verbundene Komponenten aus einem Binärbild.
- Ausführliche Beschreibung: Führt eine Komponenten-Analyse auf `binary_img` durch, legt eine leere Maske an und kopiert nur Objekte mit einer Fläche ab `min_area` hinein. Unterdrückt damit Krümel oder Rauschen und gibt das bereinigte Binärbild zurück.

## calculate_edge_sum(image)
- Kurzbeschreibung: Ermittelt die Anzahl der Kantenpixel einer Maske und liefert Kanten- sowie Binärbild zurück.
- Ausführliche Beschreibung: Konvertiert Farbbilder nach Grau, binarisiert mit Schwellwert 1 und glättet die Maske per Öffnen/Schließen. Wendet Canny-Kanten an, zählt die nicht-null Kantenpixel als Komplexitätsmaß und gibt `(edge_sum, edges, binary)` zurück, wobei `binary` für weitere Prüfungen genutzt wird.

## create_edge_report(image_data, output_file="complexity_report.png")
- Kurzbeschreibung: Erstellt einen Vergleichsbericht mit Kantenbildern vor und nach Artefaktfilterung.
- Ausführliche Beschreibung: Erwartet eine Liste von Tupeln `(pfad, edges_original, edges_clean)`, wählt bis zu fünf Beispiele aus und plottet sie in zwei Zeilen: oben Originalkanten, unten bereinigte Kanten. Speichert die Übersicht als PNG unter `output_file` und meldet den Speicherort.

## run_complexity_check(sorted_dir)
- Kurzbeschreibung: Prüft Bilder aus „Normal“ und „Bruch“ auf zu geringe oder zu hohe Kantenkomplexität und verschiebt sie ggf. nach „Rest“.
- Ausführliche Beschreibung: Lädt Bilder, berechnet die Kantenanzahl mit `calculate_edge_sum` und trifft eine erste Entscheidung: unter `MIN_EDGE_SUM` gilt als Fragment (verschieben), über `MAX_EDGE_SUM` folgt ein zweiter Durchlauf mit `remove_small_artifacts` und erneuter Kantenzählung. Liegt auch der bereinigte Wert über dem Limit, wird das Bild als „Chaos“ nach Rest verschoben, sonst behalten. Zählt verschobene sowie gerettete Bilder und berichtet den Status.
# rest.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `remove_small_artifacts(binary_img, min_area)` | Entfernt kleine Komponenten aus einem Binärbild. | Führt `connectedComponentsWithStats` auf der Maske aus, erzeugt eine leere Maske und kopiert nur die Komponenten mit Fläche ≥ `min_area` hinein. Unterdrückt Krümel/Rauschen und liefert die bereinigte Maske zurück. |
| `calculate_edge_sum(image)` | Berechnet Kantenanzahl und liefert Kanten- und Binärbild. | Wandelt das Eingabebild bei Bedarf nach Grau, binarisiert mit Schwellwert 1, glättet per Öffnen/Schließen, wendet Canny-Kanten an und zählt deren Nicht-Null-Pixel als `total_edge_length`. Gibt `(edge_sum, edges, binary)` zurück, wobei `binary` Grundlage für spätere Artefaktfilterung ist. |
| `create_edge_report(image_data, output_file="complexity_report.png")` | Visualisiert Kanten vor/nach Artefaktfilterung für bis zu fünf Beispiele. | Erwartet eine Liste von `(pfad, edges_original, edges_clean)`. Baut eine 2×N-Plotcollage: oben ursprüngliche Kantenbilder, unten bereinigte. Beschriftet Titel mit Dateiname und speichert die PNG unter `output_file`, danach schließt die Figure. |
| `run_complexity_check(sorted_dir)` | Prüft Bilder aus „Normal“ und „Bruch“ auf zu geringe oder zu hohe Kantenkomplexität und verschiebt nach „Rest“, falls nötig. | Legt `Rest`-Zielordner an, iteriert über Bilder in „Normal“/„Bruch“ und berechnet `edge_sum` via `calculate_edge_sum`. Fälle: (A) `edge_sum < MIN_EDGE_SUM` → Fragment, sofort nach Rest verschieben. (B) `edge_sum > MAX_EDGE_SUM` → Artefakt-Filter: `remove_small_artifacts` auf der Binärmaske, erneute Canny-Kanten und `clean_edge_sum`. Liegt auch dieser Wert über dem Limit, als „Chaos“ nach Rest verschieben; sonst behalten. Hält Zähler für verschobene/gerettete Bilder und gibt Statusmeldungen aus. |

Alle Funktionen des Skripts sind in der Tabelle enthalten.***
