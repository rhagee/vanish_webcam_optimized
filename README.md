# Naruto Vanish for your webcam

**English** · [Italiano](#italiano)

The video is processed on your computer and is never recorded or sent to a
server. The preview window and the virtual webcam contain only the video and
the effect, with no menus, instructions or other text overlaid.

## Credits

Based on the original project by **Nicola Auriemma**, who created the webcam
app, the Naruto Vanish effect, the log and the audio. This version is reduced
to the Naruto mode only and optimized so it doesn't slow down the webcam.

## Getting started

1. Double-click `START.bat`.
2. On the first run, step out of the frame: after 3 seconds the app captures
   the background for 3 seconds and saves it. On later runs it reuses the
   last chosen background, with no waiting.
3. Join your palms in front of the webcam, fingers side by side, and **hold
   them together for a moment** (0.2 s): your body vanishes in a puff of
   smoke, and a log appears, falls and stays visible until you reappear.

`assets/vanish.MP3` plays when the vanish starts. No microphone is needed.

## Backgrounds

Next to the preview, the **Backgrounds** window opens:

- **Capture background**: 3 seconds to step out of the frame, then 3 seconds
  of capture. The background is saved as a PNG.
- **Upload image**: pick any image (PNG, JPG, BMP, WEBP); it is copied into
  the library and fitted to the webcam resolution.
- Click a thumbnail to use it. The choice is remembered on the next launch.
- **Fixed background** (`P`): like Google Meet's virtual background, while you
  are visible you are cut out and shown over the chosen background, so the
  vanish is perfectly consistent even with an uploaded image. It costs a few
  milliseconds per frame; the choice is remembered.

Backgrounds are saved in the `backgrounds/` folder and the choice in
`settings.json`, next to the program.

## Controls

Global shortcuts, they work even when the window is not focused (e.g. during a
call in the browser):

- `Ctrl+Alt+R`: make the person reappear.
- `Ctrl+Alt+V`: trigger the effect manually.
- `Ctrl+Alt+B`: capture a new background.

With the preview window focused:

- `B` or `SPACE`: capture a new background.
- `U`: upload an image as background.
- `P`: toggle the fixed background.
- `R` / `V`: reappear / vanish.
- `D`: open or close the **Hand debug** window.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit the program and release the webcams.

You can change webcam, resolution and FPS:

```powershell
START.bat --camera 1 --width 1280 --height 720 --fps 30
```

## Hand debug

Launch with `--debug` (or press `D`) to open the **Hand debug** window,
separate from the video: it shows the hand landmarks found on the full frame
(green) and in the zoomed crop (orange), the value of each check and the
verdict of the current algorithm (**OK** green / **NO** red).

In debug mode the effect **does not play**: when the algorithm recognizes the
pose, the window freezes on that frame and stays frozen until you answer
**YES** (it was the Naruto pose) or **NO** (false positive), with the buttons
or the `Y` / `N` keys.

- Click on the image: freeze the frame right away (useful for correct poses
  the algorithm misses).
- Right-click: freeze in 3 seconds, so your hands are free for the pose.
- Each answer saves the image and the data in `debug/dataset.jsonl`, useful
  for improving recognition.

## Virtual webcam and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically
tries to open **OBS Virtual Camera**. The terminal shows the device name. To
explicitly require the virtual webcam use `START_VIRTUAL.bat`. In the
receiving app choose **OBS Virtual Camera**. OBS's virtual camera must be
stopped while this app is driving it.

To use the effect in an OBS scene, run `START_OBS.bat`, add a **Window
Capture** source choosing **Webcam - Preview**, then start the virtual camera
from OBS. Alternatively you can use
[Unity Capture](https://github.com/schellingb/UnityCapture) as a separate
virtual video source:

```powershell
START_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has a single virtual webcam; don't use it as both OBS's input and output
at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Visual assets

The [log](assets/substitution_log.png) is an original illustration generated
with ImageGen using the supplied photo as a visual reference. The smoke is
drawn in real time by `naruto_effect.py` so it follows the person's position
and scales with the video. We also looked at a
[CC0 smoke sprite](https://opengameart.org/content/smoke-sprite-sheet), but
the procedural animation fits dynamic placement better without an extra
asset. No frames or assets from the anime are included in the project.

---

<a id="italiano"></a>

# Naruto Vanish per webcam

[English](#naruto-vanish-for-your-webcam) · **Italiano**

Il video viene elaborato sul computer e non viene registrato né inviato a un
server. La finestra di anteprima e la webcam virtuale contengono solo il video
e l'effetto, senza menu, istruzioni o altre scritte sovrapposte.

L'interfaccia dell'app è in inglese; qui sotto i nomi di finestre e bottoni
sono riportati come appaiono nell'app.

## Crediti

Basato sul progetto originale di **Nicola Auriemma**, che ha creato l'app
webcam, l'effetto Naruto Vanish, il tronco e l'audio. Questa versione è
semplificata alla sola modalità Naruto e ottimizzata per non rallentare la
webcam.

## Avvio

1. Fai doppio clic su `START.bat`.
2. Al primo avvio esci dall'inquadratura: dopo 3 secondi l'app cattura lo
   sfondo per 3 secondi e lo salva. Agli avvii successivi riusa lo sfondo
   scelto l'ultima volta, senza attese.
3. Unisci i palmi davanti alla webcam, con le dita affiancate, e **tienili
   uniti per un attimo** (0,2 s): il corpo svanisce in una nube di fumo, compare
   un tronco che cade e resta visibile finché non riappari.

Quando parte la scomparsa viene riprodotto `assets/vanish.MP3`. Non serve un
microfono.

## Sfondi

Accanto all'anteprima si apre la finestra **Backgrounds**:

- **Capture background**: 3 secondi per uscire dall'inquadratura, poi 3
  secondi di cattura. Lo sfondo viene salvato come PNG.
- **Upload image**: scegli una qualsiasi immagine (PNG, JPG, BMP, WEBP);
  viene copiata nella libreria e adattata alla risoluzione della webcam.
- Clicca una miniatura per usarla. La scelta viene ricordata alla riapertura.
- **Fixed bg** (`P`, sfondo fisso): come lo sfondo virtuale di Google Meet,
  mentre sei visibile vieni ritagliato e mostrato sopra lo sfondo scelto, così
  la sparizione è perfettamente coerente anche con un'immagine caricata. Costa
  qualche millisecondo per fotogramma; la scelta viene ricordata.

Gli sfondi sono salvati nella cartella `backgrounds/` e la scelta in
`settings.json`, accanto al programma.

## Comandi

Scorciatoie globali, funzionano anche se la finestra non è in primo piano
(es. durante una call nel browser):

- `Ctrl+Alt+R`: fa riapparire la persona.
- `Ctrl+Alt+V`: avvia manualmente l'effetto.
- `Ctrl+Alt+B`: cattura un nuovo sfondo.

Con la finestra dell'anteprima in primo piano:

- `B` oppure `SPAZIO`: cattura un nuovo sfondo.
- `U`: carica un'immagine come sfondo.
- `P`: attiva o disattiva lo sfondo fisso.
- `R` / `V`: riappari / svanisci.
- `D`: apre o chiude la finestra **Hand debug**.
- `F`: attiva o disattiva lo schermo intero.
- `Q` oppure `ESC`: chiude il programma e libera le webcam.

Puoi cambiare webcam, risoluzione e FPS:

```powershell
START.bat --camera 1 --width 1280 --height 720 --fps 30
```

## Debug mani

Avvia con `--debug` (oppure premi `D`) per aprire la finestra **Hand debug**,
separata dal video: mostra i punti delle mani trovati sul fotogramma intero
(verde) e nel ritaglio ingrandito (arancione), i valori di ogni controllo e il
verdetto dell'algoritmo attuale (**OK** verde / **NO** rosso).

In modalità debug l'effetto **non parte**: quando l'algoritmo riconosce la
posa, la finestra si blocca su quel fotogramma e resta ferma finché non
rispondi **YES** (era la posa Naruto) o **NO** (falso positivo), con i bottoni
oppure con i tasti `Y` / `N` (funziona anche `S`).

- Clic sull'immagine: blocca subito il fotogramma (utile per le pose giuste
  che l'algoritmo non riconosce).
- Clic destro: blocca tra 3 secondi, così hai le mani libere per la posa.
- Ogni risposta salva l'immagine e i dati in `debug/dataset.jsonl`, utili per
  migliorare il riconoscimento.

## Webcam virtuale e OBS

Se è installato [OBS Studio](https://obsproject.com/), l'app prova ad aprire
automaticamente **OBS Virtual Camera**. Il terminale indica il nome del
dispositivo. Per richiedere esplicitamente la webcam virtuale usa
`START_VIRTUAL.bat`. Nell'app destinataria scegli **OBS Virtual Camera**.
La fotocamera virtuale di OBS deve essere ferma mentre questa app la produce.

Per usare l'effetto in una scena OBS, avvia `START_OBS.bat`, aggiungi una
sorgente **Cattura finestra** scegliendo **Webcam - Preview**, poi avvia la
fotocamera virtuale da OBS. In alternativa puoi usare
[Unity Capture](https://github.com/schellingb/UnityCapture) come sorgente video
virtuale distinta:

```powershell
START_VIRTUAL.bat --virtual-backend unitycapture
```

OBS ha una sola webcam virtuale; non usarla simultaneamente come ingresso e
uscita di OBS. Vedi la [documentazione di pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Asset visivi

Il [tronco](assets/substitution_log.png) è un'illustrazione originale generata
con ImageGen usando la foto fornita come riferimento visivo. Il fumo è disegnato
in tempo reale da `naruto_effect.py` affinché segua la posizione della persona e
si ridimensioni con il video. Abbiamo esaminato anche uno
[sprite di fumo CC0](https://opengameart.org/content/smoke-sprite-sheet), ma
l'animazione procedurale si adatta meglio al posizionamento dinamico senza un
asset aggiuntivo. Nessun fotogramma o asset dell'anime è incluso nel progetto.
