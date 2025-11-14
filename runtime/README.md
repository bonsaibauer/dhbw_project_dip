Bundled Python Installer
========================

Dieses Verzeichnis enthält den offiziellen Windows-Installer `python-3.11.9-amd64.exe`.
`setup_and_run.bat` nutzt diesen Installer automatisch, um – falls nötig – eine
vollständige Python-Installation nach `runtime\python311\` zu entpacken (inklusive
`tkinter`). Beim nächsten Start wird die vorhandene Installation wiederverwendet.

Die durch den Installer erzeugten Dateien (`runtime\python314\…`) werden über die
`.gitignore` ausgeschlossen, sodass nur der Installer selbst versioniert werden
muss.
