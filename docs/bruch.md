# bruch.py

## check_local_variance(distances, window_size=20)
- Kurzbeschreibung: Berechnet die lokale Standardabweichung über ein gleitendes Fenster auf den Radien entlang einer Kontur.
- Ausführliche Beschreibung: Polstert die Radien zyklisch, schiebt ein Fenster der Länge `window_size` über alle Punkte und sammelt für jede Position die Standardabweichung. Liefert ein Array der lokalen Varianzwerte, das später unruhige Bereiche einer Kontur sichtbar macht.

## get_radial_profile(contour)
- Kurzbeschreibung: Bestimmt den Schwerpunkt einer Kontur und misst die Distanz jedes Konturpunkts zu diesem Zentrum.
- Ausführliche Beschreibung: Berechnet aus den Bildmomenten den Schwerpunkt (cx, cy). Für jeden Konturpunkt wird der euklidische Abstand zu diesem Zentrum ermittelt und als Array zurückgegeben. Liefert zusätzlich das Zentrum; wenn die Konturfläche null ist, gibt die Funktion ein leeres Profil und (0,0) zurück.

## count_peaks(values, window=10, min_dist=200)
- Kurzbeschreibung: Zählt ausgeprägte lokale Maxima in einer Radiuskurve und fasst nahe Peaks zusammen.
- Ausführliche Beschreibung: Glättet die Werte per gleitendem Mittel, sucht lokale Maxima in einem gepolsterten Array und filtert Plateaus heraus. Anschließend sortiert sie die Kandidaten nach Höhe und führt Non-Maximum-Suppression mit zyklischem Abstand `min_dist` durch, sodass nur deutlich getrennte Ecken gezählt werden. Gibt die Anzahl der erkannten Peaks zurück.

## analyze_snack_geometry(image)
- Kurzbeschreibung: Analysiert ein zugeschnittenes Snack-Bild und klassifiziert es als „Normal“, „Bruch“ oder „Rest“ anhand von Konturmerkmalen.
- Ausführliche Beschreibung: Erstellt aus dem Grauwertbild eine Maske, reinigt sie morphologisch und extrahiert Außen- und Innenkonturen. Prüft die Außenform auf Einschnürungen, abrupte Radiuswechsel und hohe Radialvarianz; erkennt Innenfenster und bewertet deren Anzahl sowie Eckenanzahl je Fenster mit `count_peaks`. Liefert die Kategorie und einen Begründungstext zurück.

## create_visual_report(image_paths, output_file="analysis_report.png")
- Kurzbeschreibung: Generiert einen vierzeiligen Matplotlib-Bericht für bis zu fünf Bilder mit Masken, Außenradien, Varianz und Innenprofilen.
- Ausführliche Beschreibung: Lädt die angegebenen Bilder, baut Masken, extrahiert Außen- und Innenkonturen und bereitet Radial- sowie Varianzkurven auf. Plottet für jedes Bild Maske, geglätteten Außenradius mit Schwelle, Varianzverlauf und normalisierte Innenprofile (mit Eckenmarkierung). Speichert die Übersicht als PNG unter `output_file`.

## sort_images(source_dir, target_dir)
- Kurzbeschreibung: Sortiert alle Bilder aus `source_dir` in die Klassen „Normal“, „Bruch“ und „Rest“ und kopiert sie nach `target_dir`.
- Ausführliche Beschreibung: Legt Zielordner an, läuft rekursiv durch den Quellpfad und liest nur Bilddateien. Für jedes Bild ruft die Funktion `analyze_snack_geometry` auf, entscheidet die Zielklasse, kopiert die Datei dorthin (mit ggf. Präfix aus dem Quellunterordner) und zählt die Resultate. Gibt Fortschritt über die Konsole aus und protokolliert erkannte Brüche mit Begründung.
# bruch.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `check_local_variance(distances, window_size=20)` | Berechnet die lokale Standardabweichung entlang einer Radiusfolge. | Polstert das Radius-Array zyklisch, schiebt ein Fenster der Länge `window_size` über jeden Index und sammelt die Standardabweichung der Werte im Fenster. Liefert ein Array gleicher Länge, das lokale Unruhe der Kontur zeigt und später als Indikator für Bruchkandidaten dient. |
| `get_radial_profile(contour)` | Ermittelt Schwerpunkt und Radialprofil einer Kontur. | Berechnet Bildmomente, bricht ab wenn Fläche 0 ist, bestimmt Schwerpunkt `(cx, cy)` und misst für jeden Konturpunkt den euklidischen Abstand zum Schwerpunkt. Gibt das Radius-Array und das Schwerpunkt-Tupel zurück; liefert `(None, (0,0))` bei degenerierter Kontur. |
| `count_peaks(values, window=10, min_dist=200)` | Zählt signifikante Peaks in einer Kurve und fasst nahe Maxima zusammen. | Glättet die Werte mittels gleitendem Mittel (`window`), sucht lokale Maxima in einem gepolsterten Array, filtert Plateaus heraus und sortiert Kandidaten nach Höhe. Führt Non-Maximum-Suppression mit zyklischem Abstand `min_dist` durch, sodass nur voneinander getrennte Ecken übrigbleiben. Rückgabe ist die Anzahl der finalen Peaks. |
| `analyze_snack_geometry(image)` | Klassifiziert ein Bild basierend auf Außen- und Innenkontur als „Normal“, „Bruch“ oder „Rest“. | Wandelt in Grau, erzeugt und reinigt eine Maske, extrahiert Außenkontur und deren Radien. Prüft auf Einschnürungen, starke Gradienten und hohe lokale Varianz (Phase 1). Sucht anschließend Innenfenster (Phase 2), filtert nach Lage und Fläche, zählt sie und ermittelt pro Fenster Ecken via `count_peaks` und `MIN_PEAK_DISTANCE`. Liefert Kategorie plus Begründungstext zurück, inklusive Abbruchpfade bei fehlenden Konturen. |
| `create_visual_report(image_paths, output_file="analysis_report.png")` | Erstellt einen vierzeiligen Matplotlib-Bericht für bis zu fünf Bilder. | Lädt jedes Bild, erzeugt Masken, Außen- und Innenkonturen. Plottet pro Bild: Maske, geglätteten Außenradius mit Schwelle, Radialvarianz mit Schwellenlinie und normalisierte Innenprofile, farblich hervorgehoben bei zu vielen Ecken. Speichert die Collage unter `output_file` und schließt die Figure. |
| `sort_images(source_dir, target_dir)` | Sortiert Bilder in `source_dir` in die Klassen „Normal“, „Bruch“, „Rest“ und kopiert sie nach `target_dir`. | Entfernt ein altes Zielverzeichnis, legt Unterordner an, läuft rekursiv über alle Bilddateien und liest sie. Ruft `analyze_snack_geometry` auf, setzt die Kategorie, baut einen ggf. Unterordnerpräfix in den Dateinamen ein, kopiert in den Zielordner und führt Statistik über Klassenverteilung. Loggt Bruch-Fälle inklusive Grund und fasst am Ende die Zählung zusammen. |

Alle oben aufgeführten Funktionen sind im Skript vorhanden; weitere Funktionen gibt es nicht.***
