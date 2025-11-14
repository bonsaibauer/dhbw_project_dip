# Programmentwurf - Abgabe
Beachten Sie die auf dieser Folie angegebenen Vorgaben zum Programmentwurf.
Je Gruppe ist nur eine Abgabe notwendig.
Der Programmentwurf ist bis 21.11.25, 23:59 Uhr über Moodle abzugeben.
Der abgegebene Quellcode muss ohne Aufwand lauffähig sein.
Ordner für Ergebnisbilder sind automatisch zu erstellen.
Grundsätzlich sind nur die Standard Python Bibliotheken und die für die Laborversuche vorgegebenen Bibliotheken zu verwenden. Ausnahmen können nach Rücksprache mit dem Dozenten gewährt werden.
Die zur Bearbeitung bereitgestellten Daten sind nicht mit anzugeben. Nur Quellcode ohne Kommentare und die Dokumentation sind abzugeben.

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

# Git Verwaltung

## Repository herunterladen

Um das Projekt lokal herunterzuladen, klone das Git-Repository:

```bash
git clone https://github.com/bonsaibauer/dhbw_project_dip.git
cd dhbw_project_dip
```

## Standard-Git-Befehle

### Status prüfen
```bash
git status
```

### Änderungen hinzufügen
```bash
git add .
# oder einzelne Datei:
# git add pfad/zur/datei
```

### Commit erstellen
```bash
git commit -m "Beschreibung der Änderung"
```

### Änderungen zum Remote pushen
```bash
git push origin main
```

### Branch erstellen & wechseln
```bash
git checkout -b feature/mein-branch
```

### Änderungen vom Remote laden
```bash
git pull origin main
```

## Pull Request erstellen

1. Änderungen pushen  
2. Auf GitHub das Repository öffnen:  
   https://github.com/bonsaibauer/dhbw_project_dip  
3. Einen **Pull Request (PR)** für deinen Branch erstellen  
4. Änderungen und Zweck kurz beschreiben

## Lokale Ausfuehrung

Das Repository enthaelt ein Windows-Skript, das den mitgelieferten Python-Installer (`runtime\python-3.11.9-amd64.exe`) verwendet, um bei Bedarf eine komplette Python-Umgebung in `runtime\python311\` zu installieren (inkl. `tkinter`). Anschliessend installiert es alle Abhaengigkeiten und startet direkt das Tkinter-Dashboard; die Pipeline läuft dann im Hintergrund und alle Logs erscheinen im UI:

```powershell
setup_and_run.bat
```

Alle Pakete werden lokal unter `runtime\python311\Lib\site-packages\` abgelegt, sodass keine globale Python-Installation notwendig ist. Unmittelbar nach dem Start erscheint das Tkinter-Dashboard; waehrend im Hintergrund die Pipeline ausgefuehrt wird, siehst du den Log-Live-Stream und den aktuellen Status. Sobald die Berechnungen fertig sind, fuellen sich die Klassen-Tabs automatisch und du kannst wie gewohnt Detailansichten (inkl. Masken- und Zwischenschritten) oeffnen.
