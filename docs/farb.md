# farb.py

## detect_defects(image, spot_threshold=43, debug=False)
- Kurzbeschreibung: Ermittelt dunkle oder verbrannte Stellen im Snackbild und liefert Flächenmaß sowie Konturen zurück.
- Ausführliche Beschreibung: Erstellt eine Objektmaske, beschneidet Ränder, kombiniert einen Blackhat-Kontrastfilter mit einem HSV-Farbfilter für niedrige Werte/Sättigung, verknüpft die Masken und bereinigt sie morphologisch. Findet Konturen oberhalb einer Mindestfläche, summiert deren Bereiche und markiert das Bild als fehlerhaft, wenn die Gesamtfläche `spot_threshold` überschreitet. Gibt ein Ergebnis-Dictionary mit Flag, Fläche und Konturen aus.

## run_color_check(sorted_dir)
- Kurzbeschreibung: Prüft Bilder der Klasse „Normal“ auf Farbfehler und verschiebt auffällige Exemplare nach `Farbfehler`.
- Ausführliche Beschreibung: Legt im gegebenen `sorted_dir` einen Ordner `Farbfehler` an, iteriert durch alle Bilder in „Normal“, liest jedes Bild und ruft `detect_defects` mit strenger Schwelle auf. Markiert gefundene Defekte mit Konturen und erweiterten Kreisen im Bild, speichert sie im `Farbfehler`-Ordner und löscht das Original. Zählt verschobene Bilder und gibt eine Zusammenfassung auf der Konsole aus.
# farb.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `detect_defects(image, spot_threshold=43, debug=False)` | Findet dunkle/verbrannte Stellen und misst ihre Gesamtfläche. | Konvertiert das Bild in Grau, erstellt eine Objektmaske, schrumpft sie (Erosion) und ermittelt Blackhat-Regionen mit elliptischem Kernel. Schwellt Kontraste und kombiniert sie mit einer HSV-Selektion für niedrige Helligkeit/Sättigung. Beschränkt Treffer auf den Objektbereich, bereinigt sie per Öffnen, findet Konturen, filtert nach Mindestfläche 35 px², summiert die Flächen und setzt `is_defective`, wenn die Summe `spot_threshold` übersteigt. Gibt ein Dict mit Flag, Fläche und Konturen zurück. |
| `run_color_check(sorted_dir)` | Prüft Bilder der Klasse „Normal“ auf Farbfehler und verschiebt erkannte Fälle in `Farbfehler`. | Legt den Ordner `Farbfehler` im `sorted_dir` an, iteriert über alle Bilder in „Normal“, lädt sie und ruft `detect_defects` mit engerer Schwelle (`spot_threshold=20`) auf. Zeichnet gefundene Konturen und umschließende Kreise in das Bild, speichert das markierte Bild im Fehlerordner und entfernt das Original. Zählt verschobene Bilder und meldet den Abschluss. |

Alle im Skript vorhandenen Funktionen sind oben erfasst.***
