# Code- und Projektstruktur

Dieses Dokument fasst die Dateiorganisation, Namenskonventionen und Verantwortlichkeiten zusammen. Die Tabelle enthält die wichtigsten Module/Folders mit Kurzbeschreibung.

## Gesamtübersicht

```
dhbw_project_dip/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ pipeline.py
│  ├─ models/
│  ├─ steps/
│  ├─ ui/
│  └─ utils/
├─ data/
│  ├─ Images/
│  ├─ Masks/
│  └─ image_anno.csv
├─ output/
│  ├─ classified/
│  ├─ inspection/
│  └─ reports/
├─ runtime/ (lokale Python-Umgebung)
├─ run_pipeline.py
├─ setup_and_run.bat
├─ README.md
└─ docs/
```

## Modul- und Aufgabenübersicht

| Pfad/Datei | Beschreibung |
|------------|--------------|
| `app/config.py` | Zentrale Konstanten (Datenpfade, Klassenzuordnung, Tag-Mappings). |
| `app/pipeline.py` | Orchestriert den kompletten Ablauf, konfiguriert Segmenter/Klassifikator, sammelt Artefakte. Enthält `PipelineOptions`. |
| `app/models/records.py` | Datenklassen (`ProcessRecord`) für den Viewer. |
| `app/models/tree_params.py` | Serialisierte Entscheidungsbaum-Parameter (Kinder, Thresholds, Klassen). |
| `app/steps/dataset_loader.py` | Liest `image_anno.csv`, liefert `ImageRecord`-Liste. |
| `app/steps/background_segmentation.py` | Entfernt Hintergrund, liefert Maske & Crops. Nutzt die einstellbaren Parameter. |
| `app/steps/feature_extraction.py` | Berechnet Form-/Farb-/Texturmerkmale (`FEATURE_NAMES`). |
| `app/steps/classifier.py` | `RuleBasedClassifier` traversiert den Entscheidungsbaum ohne scikit-learn-Abhängigkeit. |
| `app/steps/evaluator.py` | Accuracy, Classification Report, Confusion Matrix. |
| `app/steps/result_writer.py` | Exportiert Crops in Klassenordner sowie Fehlklassifikationen nach `Falsch`. |
| `app/ui/main.py` | Startet Tkinter-Viewer, verwaltet Hintergrund-Threads für neue Läufe. |
| `app/ui/viewer.py` | Dashboard mit Tabs, Detailpanel, Parametersteuerung, Log-Panel. |
| `app/utils/logger.py` | Einfacher Logger mit Listener-Mechanismus für die UI. |
| `run_pipeline.py` | CLI-Einstieg: führt Pipeline aus, druckt Accuracy, öffnet Viewer mit Ergebnissen. |
| `setup_and_run.bat` | Windows-Skript zur Einrichtung der Runtime und Ausführung des Dashboards. |
| `docs/` | Zusatzdokumentation (Parameter, Ablauf, Architektur). |

## Namenskonventionen & Ordner

- **`app/steps`** enthält jeweils einen bewusst kleinen, klar abgegrenzten Verarbeitungsschritt.
- **`ProcessRecord`** sammelt alles, was der Viewer braucht (Pfad, Vorhersage, Metriken, Prozessschritte).
- **`output/inspection/<Klasse>/<Dateiname>/`** beherbergt sämtliche Zwischenergebnisse je Bild.
- Neue Dokumente oder Skripte gehören nach `docs/` bzw. in eigene Unterordner, um UI/Pipeline-Code sauber zu halten.

## Erweiterungshinweise

- Neue Features oder Parameter sollten in `PipelineOptions` aufgenommen und bis zu den betreffenden Modulen durchgereicht werden.
- Für reproduzierbare Konfigurationen empfiehlt es sich, die gewählten Parameterwerte zusätzlich in `output/reports/` zu loggen.
- Tests oder Debugging-Skripte sollten in separaten Ordnern (`tests/`, `scripts/`) leben, um die Kernlogik nicht zu verfälschen.
