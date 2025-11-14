# Git-Verwaltung und lokale Ausfuehrung

## Repository herunterladen

Um das Projekt lokal herunterzuladen, klone das Git-Repository:

```bash
git clone https://github.com/bonsaibauer/dhbw_project_dip.git
cd dhbw_project_dip
```

## Standard-Git-Befehle

### Status pruefen

```bash
git status
```

### Aenderungen hinzufuegen

```bash
git add .
# oder einzelne Datei:
# git add pfad/zur/datei
```

### Commit erstellen

```bash
git commit -m "Beschreibung der Aenderung"
```

### Aenderungen zum Remote pushen

```bash
git push origin main
```

### Branch erstellen & wechseln

```bash
git checkout -b feature/mein-branch
```

### Aenderungen vom Remote laden

```bash
git pull origin main
```

## Pull Request erstellen

1. Aenderungen pushen  
2. Auf GitHub das Repository oeffnen:  
   https://github.com/bonsaibauer/dhbw_project_dip  
3. Einen **Pull Request (PR)** fuer deinen Branch erstellen  
4. Aenderungen und Zweck kurz beschreiben

## Lokale Ausfuehrung

Das Repository enthaelt ein Windows-Skript, das den mitgelieferten Python-Installer (`runtime\python-3.11.9-amd64.exe`) verwendet, um bei Bedarf eine komplette Python-Umgebung in `runtime\python311\` zu installieren (inkl. `tkinter`). Anschliessend installiert es alle Abhaengigkeiten und startet direkt das Tkinter-Dashboard; die Pipeline laeuft dann im Hintergrund und alle Logs erscheinen im UI:

```powershell
setup_and_run.bat
```

Alle Pakete werden lokal unter `runtime\python311\Lib\site-packages\` abgelegt, sodass keine globale Python-Installation notwendig ist. Unmittelbar nach dem Start erscheint das Tkinter-Dashboard; waehrend im Hintergrund die Pipeline ausgefuehrt wird, siehst du den Log-Live-Stream und den aktuellen Status. Sobald die Berechnungen fertig sind, fuellen sich die Klassen-Tabs automatisch und du kannst wie gewohnt Detailansichten (inkl. Masken- und Zwischenschritten) oeffnen.
