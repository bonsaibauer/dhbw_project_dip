# dhbw_project_dip

Dieses Repository enthaelt die vollautomatisierte Bildverarbeitungs-Pipeline inklusive Tkinter-Dashboard. Das Setup erfolgt ueber das mitgelieferte Windows-Skript und benoetigt keine globale Python-Installation.

## Projektstart

1. **Direktstart (Doppelklick)**  
   Fuehre `setup_and_run.bat` per Doppelklick im Projektstamm aus. Das Skript installiert automatisch die gebuendelte Python-Version (`runtime\python311\`), richtet alle Abhaengigkeiten ein und startet anschliessend das UI.

2. **Start aus VS Code (PowerShell-Terminal)**  
   Oeffne das integrierte Terminal in VS Code, waehle PowerShell und fuehre bei Bedarf mit gelockerter Ausfuehrungsrichtlinie:
   ```powershell
   .\setup_and_run.bat
   ```
   Im Standardfall reicht `.\setup_and_run.bat`. Auch hier uebernimmt das Skript Installation und Start vollautomatisch.

Nach erfolgreichem Setup oeffnet sich das Tkinter-Dashboard. Die Pipeline laeuft parallel, und alle Logs sowie Statusmeldungen erscheinen direkt im UI.

## Bilddaten bereitstellen

- Lege den Ordner mit den Eingangsbildern idealerweise als `data\` im Projekt-Root ab, damit die Anwendung ihn sofort laden kann.  
- Falls die Bilder an einem anderen Speicherort liegen, waehle beim Start des Dashboards den tatsaechlichen Ordnerpfad aus.  
- Ohne einen gueltigen Bilder-Ordner koennen keine Klassifizierungen gestartet werden.

