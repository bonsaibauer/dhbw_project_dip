# segmentierung.py

## run_preprocessing(image, result)
- Kurzbeschreibung: Entfernt den grünen Hintergrund, findet das Hauptobjekt und erzeugt ein perspektivisch entzerrtes, zugeschnittenes Bild.
- Ausführliche Beschreibung: Konvertiert das Bild in HSV, maskiert Grünanteile und behält das Objekt. Sucht Konturen, filtert große Flächen und erzwingt Querformat der Minimal-Box. Maskiert Außenbereiche schwarz, berechnet eine Perspektivtransformation auf ein Standardformat (600×400, anschließend Resize auf 400×400) und legt das Ergebnis unter dem Schlüssel „Result“ in der übergebenen Ergebnisliste ab. Gibt zurück, ob eine Verarbeitung stattfand.

## prepare_dataset(source_dir, target_dir)
- Kurzbeschreibung: Lädt alle Bilder aus `source_dir`, führt `run_preprocessing` aus und speichert die Ergebnisse mit gleicher Unterordnerstruktur in `target_dir`.
- Ausführliche Beschreibung: Löscht ein vorhandenes Zielverzeichnis, erstellt es neu und durchläuft rekursiv alle Dateien. Für jedes erkannte Bild ruft die Funktion die Vorverarbeitung auf; vorhandene Resultate werden unter Beibehaltung des relativen Pfads mit originalem Dateinamen abgespeichert. Zählt die verarbeiteten Bilder und meldet den Fortschritt in der Konsole.
# segmentierung.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `run_preprocessing(image, result)` | Entfernt grünen Hintergrund, findet das Hauptobjekt und erzeugt ein entzerrtes Quadratbild. | Kopiert das Bild, wandelt nach HSV, erstellt eine Grünmaske (`lower_green`–`upper_green`) und invertiert sie zur Objektmaske. Maskiert den Hintergrund schwarz, findet Konturen, filtert auf große Flächen > 30 000 px und erzwingt Querformat der Minimalumrahmung. Zeichnet die Kontur in eine Maske, schwärzt Außenbereiche, berechnet eine Perspektivtransformation auf 600×400 px mit vordefinierten Zielpunkten, warp’t das Bild und resized auf 400×400. Speichert das Ergebnis als Dict `{"name": "Result", "data": warped}` in der übergebenen Liste und gibt zurück, ob eine Verarbeitung stattfand. |
| `prepare_dataset(source_dir, target_dir)` | Führt die Vorverarbeitung für alle Bilder durch und spiegelt die Ordnerstruktur ins Ziel. | Löscht ein vorhandenes Zielverzeichnis, legt es neu an und läuft rekursiv durch `source_dir`. Erzeugt pro Unterordner den passenden Zielpfad, lädt Bilder mit üblichen Endungen, ruft `run_preprocessing` auf und speichert jedes gefundene `Result` unter demselben relativen Pfad und Dateinamen im `target_dir`. Zählt verarbeitete Bilder und meldet Start/Ende in der Konsole. |

Alle Funktionen des Skripts sind in der Tabelle erfasst.***
