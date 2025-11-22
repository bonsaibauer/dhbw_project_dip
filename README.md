# Digital Image Processing – DHBW Projekt

## 1. Umgebung & Daten vorbereiten

1. **Python-Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```
2. **Datenstruktur vorbereiten**  
   - Im Ordner `data/Images` müssen die Rohbilder liegen (Standard: Unterordner `Normal/` und `Anomaly/`).  
   - Optional zugehörige Masken in `data/Masks`.  
   - Annotationen (`data/image_anno.csv`) werden für die Validierung benötigt.  
   - Wenn andere Pfade genutzt werden sollen, müssen die Konstanten am Anfang von `main.py` entsprechend angepasst werden (Struktur bleiben: `Images/<Klasse>/<Bild>.JPG`).

## 2. Workflow ausführen

```bash
python main.py
```

## 3. Vorgehensweise / Systemarchitektur
`main.py` erwartet die Rohbilder unter `data/Images`. Ist dieser Pfad nicht vorhanden, erscheint eine Abfrage im Terminal, über die Heintz einen anderen Quellordner angeben kann; ohne Eingabe oder bei einem ungültigen Pfad bricht das Programm ab.

Beim Start legt das Skript den Ordner `output/` an und darin die Unterordner `processed/` (zugeschnittene Bilder) und `sorted/` (Ergebnis der Sortier- und Prüfmodule). Alle Zwischenergebnisse und finalen Klassenergebnisse werden dort abgelegt, daher sind für `output/` Schreibrechte erforderlich. Falls das Standardverzeichnis nicht beschreibbar ist, sollte vor dem Lauf ein alternativer, beschreibbarer Output-Pfad gesetzt werden.
