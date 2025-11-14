# Programmentwurf – Aufgabenstellung

Beachten Sie die auf dieser Folie angegebenen Vorgaben zum Programmentwurf.
Je Gruppe ist nur eine Abgabe notwendig.
Der Programmentwurf ist bis 21.11.25, 23:59 Uhr ueber Moodle abzugeben.
Der abgegebene Quellcode muss ohne Aufwand lauffaehig sein.
Ordner fuer Ergebnisbilder sind automatisch zu erstellen.
Grundsaetzlich sind nur die Standard-Python-Bibliotheken und die fuer die Laborversuche vorgegebenen Bibliotheken zu verwenden. Ausnahmen koennen nach Ruecksprache mit dem Dozenten gewaehrt werden.
Die zur Bearbeitung bereitgestellten Daten sind nicht mit anzugeben. Nur Quellcode ohne Kommentare und die Dokumentation sind abzugeben.

## Aufgabe

- Entfernen Sie den Hintergrund und schneiden Sie die Objekte aus. Wenn fuer die Auswertung hilfreich, koennen die Bilder auf eine einheitliche Groesse transformiert werden.
- Entwickeln Sie Algorithmen, um anhand der Bilddaten die zugeschnittenen Bilder in die Ordner **"Normal"**, **"Farbfehler"**, **"Bruch"** und **"Rest"** einzusortieren. Diese Klassen stellen eine Vereinfachung der Klassen in *image_anno.csv* dar. Sollte Unklarheit bei der Zuordnung bestehen, kann dies mit dem Dozenten besprochen werden. Laesst sich ein Bild mehreren Klassen zuordnen, kann eine beliebige Zuordnung gewaehlt werden.
- Des Weiteren sollen alle falsch zugeordneten Bilder in den Ordner **"Falsch"** kopiert werden. Der Dateiname ist dabei um die korrekte und die erkannte Klasse zu ergaenzen.
- Legen Sie ein Mass fuer die Symmetrie der Objekte fest und sortieren Sie die Bilder der Klasse **"Normal"** mittels eines Praefix nach diesem Mass.

## Abgabe

### Quellcode

Der abgegebene Quelltext muss nur die beste Loesung umfassen.

### Dokumentation

- Die Einleitung und der Stand der Technik koennen sehr kurz gehalten werden.
- Die verwendeten Algorithmen sind zu erlaeutern und Abbildungen mit Zwischenergebnissen zu zeigen. Zusaetzlich koennen auch verworfene Ansaetze gegenuebergestellt werden.
- Im Fazit ist das Ergebnis mittels statistischer Daten und Bildern darzulegen, falsche Klassifikationen zu diskutieren und alternative Loesungsansaetze vorzuschlagen.
- Ein Loesungsansatz kann auch sein, die vorgegebene Klassenzuordnung in Frage zu stellen, wobei dies mit geeigneten Argumenten zu belegen ist.
