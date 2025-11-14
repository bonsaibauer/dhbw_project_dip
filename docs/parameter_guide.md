# Parametersteuerung und Auswirkungen

Dieses Dokument beschreibt alle interaktiven Pipeline-Parameter im Tk-Viewer. Jeder Abschnitt erklärt, **welchen Bildverarbeitungsschritt** der Regler beeinflusst, **welche Werte sinnvoll sind** und **welchen sichtbaren Effekt** Änderungen typischerweise haben.

## Segmentierungsparameter

| Parameter | Standard | Wirkung | Einfluss auf das Bild |
|-----------|----------|--------|------------------------|
| `Blur-Kernel` (`blur_size`) | 5 (ungerade) | Kanten werden vor der Schwellenwertsuche mit einem Gauß-Filter geglättet. Nur ungerade Werte sind zulässig, weil OpenCV symmetrische Kerne erwartet. | Größere Kerne verwischen mehr Rauschen, riskieren aber, schmale Risse oder kleine Kratzer zu „zukleistern“. Kleine Werte belassen feine Details, erzeugen jedoch mehr Salz-Pfeffer-Geräusche in der Maske. |
| `Median-Kernel` (`median_kernel_size`) | 5 (ungerade) | Entfernt impulsartiges Rauschen nach der binären Schwelle. | Große Werte schließen kleine Löcher im Objekt, können aber filigrane Aussparungen entfernen. Kleine Werte lassen eventuell einzelne Pixel in der Maske stehen. |
| `Morph-Kernel` (`morph_kernel_size`) | 11 (ungerade) | Größe des elliptischen Struktur-Elements für die Morphologie (Close/Open). | Ein größerer Kernel verbindet weiter auseinanderliegende Pixel und glättet Umrisse stärker. Ein kleiner Kernel arbeitet lokaler und erhält mehr Detail. |
| `Morph-Iterationen` (`morph_iterations`) | 1 | Anzahl der Wiederholungen pro Morph-Operation. | Mehr Iterationen verstärken Close/Open entsprechend: Close füllt Lücken stärker, Open entfernt mehr Ausreißer. |
| `Morph zuerst schließen (1/0)` (`close_then_open`) | 1 (True) | Reihenfolge der Morph-Operationen. `1` = Close → Open, `0` = Open → Close. | Close→Open füllt zunächst Löcher und entfernt danach kleine Objekte – gut für zusammenhängende Objekte. Open→Close entfernt zuerst Rauschen und verbindet anschließend Flächen – hilfreich bei starkem Rauschen. |
| `Nur größte Kontur (1/0)` (`keep_largest_object`) | 1 (True) | Entscheidet, ob nach der Morphologie nur die größte zusammenhängende Region behalten wird. | Wenn `1`, verschwinden zufällige Flecken vollständig, aber mehrere echte Objekte gehen verloren. `0` belässt alle Konturen, wodurch z. B. doppelte oder überlappende Fryums sichtbar bleiben. |
| `Invert-Schwelle (L)` (`invert_threshold`) | 200 | Durchschnittswert der binären Maske (0–255), ab dem diese invertiert wird. | Höhere Werte bedeuten: nur wenn sehr viel Weiß vorliegt, wird invertiert. Senkt man den Wert, invertiert die Maske schneller – hilfreich, wenn Hintergrund/Helligkeit stark variiert. |

## Feature-Extraktionsparameter

| Parameter | Standard | Wirkung | Einfluss auf Metriken |
|-----------|----------|--------|------------------------|
| `L dunkel (<)` (`dark_threshold`) | 170 | Grenze für dunkle Pixel im L-Kanal (0 = schwarz, 255 = weiß). | Niedrige Grenzwerte klassifizieren nur sehr dunkle Bereiche als „dunkel“, hohe Werte deklarieren schon mittelhelle Bereiche als dunkel. Direkt sichtbar in `dark_fraction`. |
| `L hell (>)` (`bright_threshold`) | 210 | Grenze für helle Pixel im L-Kanal. | Senken → mehr Pixel gelten als hell; Anheben → nur echte Spitzlichter fließen in `bright_fraction` ein. |
| `Gelb b>` (`yellow_threshold`) | 150 | Grenze im b-Kanal (negativ = blau, positiv = gelb). | Größere Werte erkennen nur kräftige Gelbverschiebungen. Kleinere Werte stufen auch leicht gelbliche Stellen als „Gelbanteil“ ein. |
| `Rot a>` (`red_threshold`) | 150 | Grenze im a-Kanal (negativ = grün, positiv = rot). | Analog zu Gelb: Senken → mehr Pixel gelten als rot; Anheben → nur kräftige Rotanteile zählen. |
| `Laplacian-Kernel` (`laplacian_ksize`) | 3 (ungerade) | Fenstergröße für den Laplace-Operator, der die Textur-Variation (`laplacian_std`) misst. | Kleine Kerne (3) reagieren empfindlich auf feine Strukturen, größere Kerne glätten und heben nur grobe Texturwechsel hervor. |

## Zusammenspiel der Parameter

1. **Blur/Median/Morphologie** bestimmen, wie sauber der Hintergrund entfernt wird. Je besser die Maske, desto zuverlässiger die Form- und Farbmerkmale.
2. **Invert-Schwelle** sorgt dafür, dass die Maske immer schwarz=Hintergrund, weiß=Objekt bleibt – egal ob das Bild heller oder dunkler aufgenommen wurde.
3. **Farbschwellen** wirken ausschließlich auf die bereits maskierten Pixel. Ihren Einfluss beobachtest du direkt in den Metriken eines Datensatzes (z. B. `dark_fraction` in der Detailansicht).
4. **Laplacian-Kernel** beeinflusst nur den Textur-Score. Kombiniert mit Farb- und Formmerkmalen entscheidet der Entscheidungsbaum anschließend, in welche Klasse ein Fryum fällt.

> **Praxis-Tipp:** Passe zuerst die Segmentierung (Blur/Morph) an, bis die Masken sauber aussehen. Danach finetunest du die Schwellen für L/a/b, um Klassifikationsfehler (z. B. Farbfehler vs. Normal) zu reduzieren.
