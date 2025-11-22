# symmetrie.py

## get_symmetry_score(image_bgr)
- Kurzbeschreibung: Berechnet einen Score (0–100) für 6-fache Rotationssymmetrie eines maskierten Objekts.
- Ausführliche Beschreibung: Erzeugt aus dem Bild eine Binärmaske, bestimmt den Schwerpunkt und rotiert die Maske in 60°-Schritten um dieses Zentrum. Bildet sukzessive die Schnittmenge aller Rotationen, subtrahiert diese vom Original und quantifiziert den asymmetrischen Anteil. Der Score entspricht dem verbleibenden symmetrischen Verhältnis, begrenzt und auf zwei Nachkommastellen gerundet.

## run_symmetry_check(sorted_dir)
- Kurzbeschreibung: Bewertet alle Bilder in „Normal“ mit dem Symmetriescore und präfixiert den Dateinamen mit dem Wert.
- Ausführliche Beschreibung: Sucht im `sorted_dir` nach dem Ordner „Normal“, iteriert über die Bilddateien und berechnet für jede das Ergebnis von `get_symmetry_score`. Formatiert den Score mit führenden Nullen, setzt ihn als Präfix (`99.50_Datei.jpg`) und benennt die Datei um. Protokolliert Anzahl und Durchschnitt der Scores über die Konsole.
# symmetrie.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `get_symmetry_score(image_bgr)` | Berechnet einen 0–100 Score für 6-fache Rotationssymmetrie. | Wandelt das Bild in Grau, erzeugt eine Binärmaske mit Schwellwert 10 und prüft, ob überhaupt Objektfläche vorhanden ist. Bestimmt den Schwerpunkt, kopiert die Maske als Startkern und rotiert sie in 60°-Schritten um `(cx, cy)` mit `warpAffine`. Bildet iterativ die Schnittmenge aller Rotationen (Symmetrie-Kern), subtrahiert diesen vom Original, zählt asymmetrische Pixel, berechnet den Fehleranteil und wandelt ihn in einen Score um, der auf zwei Nachkommastellen gerundet und auf [0, 100] begrenzt wird. |
| `run_symmetry_check(sorted_dir)` | Bewertet alle Bilder in „Normal“ und versieht den Dateinamen mit dem Symmetriescore. | Prüft, ob `sorted_dir/Normal` existiert, iteriert über alle Bilddateien, berechnet für jedes `get_symmetry_score`, sammelt Scores, formatiert sie mit führenden Nullen (`{score:05.2f}`) und benennt die Datei zu `"{score}_{filename}"` um. Zählt umbenannte Dateien, berechnet den Durchschnittsscore und gibt Statusmeldungen aus. |

Alle Funktionen des Skripts sind abgedeckt.***
