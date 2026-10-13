# Naruto Vanish per webcam

Il video viene elaborato sul computer e non viene registrato né inviato a un
server. La finestra di anteprima e la webcam virtuale contengono solo il video
e l'effetto, senza menu, istruzioni o altre scritte sovrapposte.

## Crediti

Basato sul progetto originale di **Nicola Auriemma**, che ha creato l'app
webcam, l'effetto Naruto Vanish, il tronco e l'audio. Questa versione è
semplificata alla sola modalità Naruto e ottimizzata per non rallentare la
webcam.

## Avvio

1. Fai doppio clic su `AVVIA_LOCALE.bat`.
2. Al primo avvio esci dall'inquadratura: dopo 3 secondi l'app cattura lo
   sfondo per 3 secondi e lo salva. Agli avvii successivi riusa lo sfondo
   scelto l'ultima volta, senza attese.
3. Unisci i palmi davanti alla webcam, con le dita affiancate, e **tienili
   uniti per un attimo** (0,2 s): il corpo svanisce in una nube di fumo, compare
   un tronco che cade e resta visibile finché non riappari.

Quando parte la scomparsa viene riprodotto `assets/vanish.MP3`. Non serve un
microfono.

## Sfondi

Accanto all'anteprima si apre la finestra **Sfondi**:

- **Cattura sfondo**: 3 secondi per uscire dall'inquadratura, poi 3 secondi
  di cattura. Lo sfondo viene salvato come PNG.
- **Carica immagine**: scegli una qualsiasi immagine (PNG, JPG, BMP, WEBP);
  viene copiata nella libreria e adattata alla risoluzione della webcam.
- Clicca una miniatura per usarla. La scelta viene ricordata alla riapertura.

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
- `R` / `V`: riappari / svanisci.
- `D`: apre o chiude la finestra **Debug mani**.
- `F`: attiva o disattiva lo schermo intero.
- `Q` oppure `ESC`: chiude il programma e libera le webcam.

Puoi cambiare webcam, risoluzione e FPS:

```powershell
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

## Debug mani

Avvia con `--debug` (oppure premi `D`) per aprire la finestra **Debug mani**,
separata dal video: mostra i punti delle mani trovati sul fotogramma intero
(verde) e nel ritaglio ingrandito (arancione), i valori di ogni controllo e il
verdetto dell'algoritmo attuale (**OK** verde / **NO** rosso).

- Clic sull'immagine: blocca il fotogramma (clic di nuovo per riprendere).
- Clic destro: blocca tra 3 secondi, così hai le mani libere per la posa.
- Quando l'effetto parte, la finestra si blocca da sola su ciò che l'ha attivato.
- Con il fotogramma bloccato, **SI** / **NO** indica se era davvero la posa
  Naruto. Ogni risposta salva l'immagine e i dati in `debug/dataset.jsonl`,
  utili per migliorare il riconoscimento.

## Webcam virtuale e OBS

Se è installato [OBS Studio](https://obsproject.com/), l'app prova ad aprire
automaticamente **OBS Virtual Camera**. Il terminale indica il nome del
dispositivo. Per richiedere esplicitamente la webcam virtuale usa
`AVVIA_VIRTUAL.bat`. Nell'app destinataria scegli **OBS Virtual Camera**.
La fotocamera virtuale di OBS deve essere ferma mentre questa app la produce.

Per usare l'effetto in una scena OBS, avvia `AVVIA_OBS.bat`, aggiungi una sorgente
**Cattura finestra** scegliendo **Webcam - Anteprima**, poi avvia la fotocamera
virtuale da OBS. In alternativa puoi usare
[Unity Capture](https://github.com/schellingb/UnityCapture) come sorgente video
virtuale distinta:

```powershell
AVVIA_VIRTUAL.bat --virtual-backend unitycapture
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

