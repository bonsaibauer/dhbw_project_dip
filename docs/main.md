# main.py

## Hauptablauf (Schutz durch `if __name__ == '__main__':`)
- Kurzbeschreibung: Führt die komplette Snack-Analyse-Pipeline von der Vorverarbeitung bis zur Auswertung aus.
- Ausführliche Beschreibung: Setzt Quell-, Zwischen- und Zielpfade sowie den optionalen Annotationspfad. Prüft die Existenz der Rohdaten, ruft nacheinander die Teilmodule auf: `segmentierung.prepare_dataset` zum Zuschneiden, `bruch.sort_images` für die Grundklassifizierung, `rest.run_complexity_check` zur Komplexitätsprüfung, `farb.run_color_check` für Farbfehler und `symmetrie.run_symmetry_check` zum Bewerten der Symmetrie. Führt optional `ergebnis.evaluate_results` aus, falls eine Annotationsdatei vorliegt, und beendet mit einer Statusmeldung.
# main.py – Funktions-/Ablaufübersicht

| Funktion/Block | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `if __name__ == '__main__':` | Führt die komplette Pipeline von Rohdaten bis Auswertung aus. | Definiert Pfade für Rohbilder, Vorverarbeitung, Sortierung und optionale Annotation. Bricht mit Fehlermeldung ab, falls Rohdaten fehlen. Ablauf: (1) `segmentierung.prepare_dataset` erzeugt zugeschnittene Bilder. (2) `bruch.sort_images` klassifiziert grob in Normal/Bruch/Rest. (3) `rest.run_complexity_check` verschiebt chaotische/fragmentierte Fälle. (4) `farb.run_color_check` sucht Farbfehler in Normal. (5) `symmetrie.run_symmetry_check` versieht Normal-Bilder mit Symmetriescore. (6) Optional `ergebnis.evaluate_results`, wenn eine Annotations-CSV existiert. Abschließend Konsolenmeldung „Pipeline abgeschlossen.“ |

Das Skript enthält keine weiteren Funktionen außerhalb des Main-Blocks.***
