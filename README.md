# Digital Image Processing - DHBW Projekt

## 1. Umgebung & Daten vorbereiten

1. **Python-Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```
2. **Datenstruktur vorbereiten**  
   - Standard: `data/Images/Normal`, `data/Images/Anomaly` und `data/image_anno.csv` im Repo-Root.  
   - Alternativ: Beim Start von `main.py` einen eigenen `data`-Ordner Pfad mit exakt dieser Struktur angeben.  
   - Optional zugehörige Masken in `data/Masks`.  
   - Annotationen (`data/image_anno.csv`) werden für die Validierung benötigt.

## 2. Workflow ausführen

```bash
python main.py
```
