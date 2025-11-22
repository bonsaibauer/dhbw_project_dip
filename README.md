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