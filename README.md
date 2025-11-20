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

## 3. Dokumentation & Hintergründe
| Thema | Link | Beschreibung |
| --- | --- | --- |
| Pipeline-Überblick & Parameter | [docs/workflow.md](docs/workflow.md) | Gesamtfluss der 5 Stages, Artefakte und Konfigurationsquellen. |
| main.py Orchestrierung | [docs/main.md](docs/main.md) | Einstiegspunkte, STAGE_LIST und Pfad-Setup der Pipeline. |
| Stage 1: Segmentierung | [docs/01_segmentation.md](docs/01_segmentation.md) | Zuschnitt/Warp, HSV-Parameter und CLI für die Rohbilder. |
| Stage 2: Bildverarbeitung | [docs/02_image_processing.md](docs/02_image_processing.md) | Feature-Extraktion (Geometrie/Farbe) und CSV-Erstellung. |
| Stage 3: Klassifikation | [docs/03_classification.md](docs/03_classification.md) | Regel-Engine, Label-Prioritäten und Entscheidungsgrundlagen. |
| Stage 4: Sortierung | [docs/04_sorting.md](docs/04_sorting.md) | Kopierlogik der sortierten Ausgaben und Reporting. |
| Stage 5: Validierung | [docs/05_validation.md](docs/05_validation.md) | Abgleich mit Annotationen, Fehlklassifikationen und Statistik. |
| Aufgabenstellung | [docs/aufgabenstellung.md](docs/aufgabenstellung.md) | Originalbeschreibung der Aufgabe. |
| Git-Anleitung | [docs/github_anleitung.md](docs/github_anleitung.md) | Hinweise zum Arbeiten mit Git/GitHub. |
