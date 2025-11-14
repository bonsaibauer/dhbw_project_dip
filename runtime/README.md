# Eingebetteter Python-Interpreter

Das Setup-Skript (`setup_and_run.bat`) nutzt **ausschließlich**
die portable Python-Distribution `python-3.13.9-embed-amd64.zip`, die sich
im Ordner `runtime\` befinden muss.

Beim Start prüft das Skript automatisch:

1. Ob `runtime\python-3.13.9-embed-amd64.zip` vorhanden ist.
2. Ob der Inhalt bereits nach `runtime\python-3.13.9-embed-amd64\` entpackt wurde.
   Falls nicht, wird `Expand-Archive` aufgerufen und der Interpreter
   vorbereitet.
3. Anschließend wird `runtime\python-3.13.9-embed-amd64\python.exe` verwendet, um
   direkt aus der portablen Installation `pip` zu installieren, die Abhängigkeiten
   aus `requirements.txt` einzuspielen und `run_pipeline.py` auszuführen.

Damit ist keine globale Python-Installation erforderlich und das Projekt läuft
vollständig autark.
