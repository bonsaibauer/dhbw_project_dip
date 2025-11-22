# ergebnis.py

## get_true_label(raw_label)
- Kurzbeschreibung: Mappt ein CSV-Label auf die Projektkategorien „Normal“, „Bruch“, „Farbfehler“ oder „Rest“.
- Ausführliche Beschreibung: Zerlegt das Label an Kommas, normalisiert die Tokens und prüft in fester Reihenfolge auf Schlüsselwörter für Brüche, Reste/Fragmente, Farbfehler und Normal. Gibt die zugehörige Klasse zurück; nicht zuordenbare Labels fallen auf „Rest“ zurück.

## evaluate_results(sorted_dir, csv_path)
- Kurzbeschreibung: Vergleicht die sortierten Bilder mit Ground-Truth-Labels aus einer CSV, verschiebt Fehlzuordnungen und gibt eine Statistik aus.
- Ausführliche Beschreibung: Liest die CSV (`image`, `label`), baut eine Mapping-Tabelle mit aufbereiteten Dateipfaden, zählt Soll-Werte und legt den Ordner `Falsch` neu an. Durchläuft die Ergebnisordner, bereinigt Dateinamen (z.B. Symmetriepräfix), matcht sie gegen die Ground-Truth-Tabelle und wertet Treffer/Miss out. Verschiebt Fehlzuordnungen nach `Falsch` mit beschreibendem Namen, berechnet Treffergenauigkeiten pro Kategorie und Gesamt, meldet fehlende CSV-Bilder und druckt eine formatierte Zusammenfassung in die Konsole.
# ergebnis.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detaillierte Beschreibung |
| --- | --- | --- |
| `get_true_label(raw_label)` | Mappt CSV-Labels auf Projektkategorien. | Splittert das Rohlabel an Kommata, trimmt und normalisiert Tokens zu Kleinbuchstaben. Prüft in Prioritätsreihenfolge auf Schlüsselwörter für „Bruch“, anschließend „Rest“/Fragmente/sonstiges, dann „Farbfehler“, zuletzt „Normal“. Gibt die zugeordnete Kategorie zurück, fällt andernfalls auf „Rest“ zurück. |
| `evaluate_results(sorted_dir, csv_path)` | Vergleicht Sortierergebnis mit Ground-Truth aus CSV, verschiebt Fehlzuordnungen und erstellt eine Statistik. | Öffnet die CSV (`image`, `label`) mit UTF‑8, baut ein Mapping von bereinigten Pfaden (z.B. `folder/file.jpg`) auf True Labels über `get_true_label`. Initialisiert Statistikstrukturen und den Ordner `Falsch` (vorher löschen). Läuft alle Ergebnisordner der Kategorien durch, bereinigt Dateinamen (entfernt Symmetrie-Präfixe bei Zahl+Unterstrich), rekonstruiert Schlüssel und matched gegen die Ground-Truth-Map; nutzt Fallback auf Suffix-Matching. Erhöht Treffer oder Miss-Zähler, verschiebt Fehlzuordnungen nach `Falsch` mit sprechendem Namen `SOLL_{true}_IST_{folder}_{filename}`. Am Ende druckt sie eine formatierte Tabelle mit Soll-, Treffer- und Genauigkeitswerten pro Kategorie und gesamt, meldet fehlende CSV-Bilder und gibt den Speicherort der Fehlzuordnungen aus. |

Alle im Skript vorhandenen Funktionen sind in der Tabelle aufgeführt.***
