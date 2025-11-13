# Programmentwurf - Abgabe
Beachten Sie die auf dieser Folie angegebenen Vorgaben zum Programmentwurf.
Je Gruppe ist nur eine Abgabe notwendig.
Der Programmentwurf ist bis 21.11.25, 23:59 Uhr über Moodle abzugeben.
Der abgegebene Quellcode muss ohne Aufwand lauffähig sein.
Ordner für Ergebnisbilder sind automatisch zu erstellen.
Grundsätzlich sind nur die Standard Python Bibliotheken und die für die Laborversuche vorgegebenen Bibliotheken zu verwenden. Ausnahmen können nach Rücksprache mit dem Dozenten gewährt werden.
Die zur Bearbeitung bereitgestellten Daten sind nicht mit anzugeben. Nur Quellcode ohne Kommentare und die Dokumentation sind abzugeben.

## Programmentwurf – Datensatz
Laden Sie sich diesen Datensatz herunter und entpacken diesen in Ihren Projektordner. Es handelt sich dabei um einen Teil des Visual Anomaly (VisA) Datensatzes. Im Ordner **"Images"** befinden sich Ordner mit Bildern für normale und abnorme Objekte. Die Datei **"image_anno.csv"** und die Bilder im Ordner **"Masks"** dienen dazu, Ihnen zu vermitteln, welche Fehler zu erkennen sind und sind nicht maschinell auszuwerten.

## Programmentwurf – Aufgabe
- Entfernen Sie den Hintergrund und schneiden Sie die Objekte aus. Wenn für die Auswertung hilfreich, können die Bilder auf eine einheitliche Größe transformiert werden.
- Entwickeln Sie Algorithmen, um anhand der Bilddaten die zugeschnittenen Bilder in die Ordner **"Normal"**, **"Farbfehler"**, **"Bruch"** und **"Rest"** einzusortieren. Diese Klassen stellen eine Vereinfachung der Klassen in *image_anno.csv* dar. Sollte Unklarheit bei der Zuordnung bestehen, kann dies mit dem Dozenten besprochen werden. Lässt sich ein Bild mehreren Klassen zuordnen, kann eine beliebige Zuordnung gewählt werden.
- Des Weiteren sollen alle falsch zugeordneten Bilder in den Ordner **"Falsch"** kopiert werden. Der Dateiname ist dabei um die korrekte und die erkannte Klasse zu ergänzen.
- Legen Sie ein Maß für die Symmetrie der Objekte fest und sortieren Sie die Bilder der Klasse **"Normal"** mittels eines Präfix nach diesem Maß.

## Programmentwurf – Abgabe
### Quellcode
Der abgegebene Quelltext muss nur die beste Lösung umfassen.

### Dokumentation
- Die Einleitung und der Stand der Technik können sehr kurz gehalten werden.
- Die verwendeten Algorithmen sind zu erläutern und Abbildungen mit Zwischenergebnissen zu zeigen. Zusätzlich können auch verworfene Ansätze gegenübergestellt werden.
- Im Fazit ist das Ergebnis mittels statistischer Daten und Bildern darzulegen, falsche Klassifikationen zu diskutieren und alternative Lösungsansätze vorzuschlagen.
- Ein Lösungsansatz kann auch sein, die vorgegebene Klassenzuordnung in Frage zu stellen, wobei dies mit geeigneten Argumenten zu belegen ist.
