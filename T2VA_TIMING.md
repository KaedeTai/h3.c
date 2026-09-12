# T2VA timing: what the prompt controls, and what it does not

A T2VA prompt is the script — the Mandarin in it gets spoken, the English
conditions the picture silently. The question this document answers is whether
the prompt can also *stage* a clip: a beat of action, then the line, then a
reaction, instead of ten seconds of wall-to-wall speech.

It can, partly. Nineteen runs at 608×352, 243 frames (10.125 s), 3 steps,
`H3_VAE_INT8_FFN=1`, about 70 s each — roughly 22 minutes of M5 Max — say that
**the model reads the structure and ignores the numbers**, that the one lever
which really moves a boundary is where the line sits relative to the silent
beat, and that a silent head and a silent tail are not available at the same
time.

## Measure it with ASR word timestamps, not with energy

The silent beats are not silent. The model fills them with night-market
ambience, so `ffmpeg silencedetect=n=-40dB:d=0.3` finds **no silence at all** in
any of the nineteen takes, and a per-500 ms RMS envelope is above −30 dB almost
everywhere. What is absent is *speech*, and only a recogniser can see that.
Every number below is the first and last word timestamp from faster-whisper
`small` with `word_timestamps=True`.

That the quiet beats carry ambience rather than digital silence is itself the
good news: nothing has to be dubbed back in later.

## 1. With no beat structure the model talks end to end, and invents

The baseline prompt — the line, then a static English scene description — is
what most people write first, and it is the worst case. A 19-character line
over a 10.125 s clip came back with speech from 1.44 s to 7.16 s and an opening
the script never contained: 「反正这个辣椒和煎得超好吃…」. A 39-character line
filled 0.00–9.12 s. The model has no silence token and its prior for a
short-form clip is continuous speech, so a line shorter than the clip is padded
rather than placed.

## 2. The numbers in the prompt are not read as durations

Asking for a different number of seconds does not move the boundary. Same
prompt otherwise, 19-character line:

| asked for | speech starts | speech ends | quiet tail |
|---|---|---|---|
| first **2** seconds silent, last 3 laughing | 4.98 s | 9.04 s | 1.08 s |
| first **3** seconds silent, last 3 laughing | 5.00 s | 9.06 s | 1.06 s |
| beats given as absolute stamps, `0 s → 3 s → 7 s → 10 s` | 5.30 s | 9.04 s | 1.08 s |

and the tail instruction is even more inert:

| asked for | quiet tail measured |
|---|---|
| last **1** second laughing | 0.93 s |
| last **3** seconds laughing | 1.06 s |
| last **5** seconds laughing | 1.09 s |

Three different numbers, one answer. Absolute timestamps are no better than
durations. What the model took from all six prompts is the same thing: *there
is a quiet beat and then talking*.

## 3. With a trailing silent beat the speech is right-aligned, and the head is
set by the line's length

Hold the beat structure fixed ("first three seconds silent … last three seconds
laughing", seed 42) and vary only how many characters the line has:

| line | speech | duration | silent head | rate |
|---|---|---|---|---|
| 9 characters | 5.96 – 8.68 s | 2.72 s | **5.96 s** | 3.3 char/s |
| 19 characters | 5.00 – 9.06 s | 4.06 s | **5.00 s** | 4.7 char/s |
| 29 characters | 0.00 – 9.06 s | 9.06 s | **0 s** | padded |
| 39 characters | 0.00 – 9.16 s | 9.16 s | **0 s** | 4.3 char/s |

Two things are going on. The end of the speech is pinned about **one second
before the clip ends** — 0.93 to 1.66 s across every take in this document that
has a trailing quiet beat — and the line is laid down ending there. So the
silent head is what is left over: `head ≈ clip − 1 s − characters / 4.5`.

The second thing is the failure mode. Past roughly **45% of the clip's speech
capacity** the head does not shrink gracefully, it collapses to zero and the
model starts padding: the 29-character line should have wanted 6.4 s and left
2.7 s of head, and instead it filled the whole clip and opened with 「相信他這個
泡泡泡泡品的店不知道…」. The 39-character line happens to fit the full width
honestly, which is why it is clean but has no head at all.

Speaking rate is 3.3–5.1 characters a second across all nineteen takes,
clustering at about 4.5. Short lines drawl; long ones do not speed up, they
overflow.

## 4. Seeds move a boundary by about ±0.3 s

The same 19-character line and beat structure at three seeds:

| seed | speech starts | speech ends |
|---|---|---|
| 42 | 5.00 s | 9.06 s |
| 7 | 4.60 s | 8.46 s |
| 3 | 5.24 s | 9.22 s |

A spread of 0.64 s on the start. The gap between a 9-character and a
19-character line (0.96 s) is larger than that, so §3 is a real effect and not
seed noise — but only just, and any single take can sit a third of a second
either side of where the rule puts it.

## 5. Front-loading the speech is what buys a long tail — and the wording does
it, not the number

If the speech comes first and the quiet beat last, the tail opens up. Two
phrasings of the same idea, crossed against seeds:

**A.** "…says 「line」 right away, in the first four seconds. For the remaining
six seconds she says nothing at all and only laughs and grins at the camera, no
more words, silent."

**B.** "…says 「line」 immediately as the shot opens. Then she is silent for the
rest of the shot, laughing at the camera with her mouth closed, no speech, no
words, for seven seconds."

| prompt | seed | speech | quiet tail | transcript |
|---|---|---|---|---|
| A | 42 | 0.00 – 7.22 s | 2.91 s | clean |
| A | 7 | 0.00 – 8.50 s | 1.62 s | **padded** with invented words |
| B | 7 | 0.00 – 3.76 s | **6.37 s** | clean |
| B | 42 | 0.00 – 3.66 s | **6.46 s** | clean |
| B | 3 | 0.00 – 3.86 s | **6.27 s** | clean |

B is stable to ±0.1 s across three seeds and says exactly the nineteen
characters it was given, three times out of three. A drifts by 1.3 s and pads on
one of two seeds. The two prompts ask for the same shot; what separates them is
how the silence is written:

- B says **"for the rest of the shot"**; A gives a number. Consistent with §2 —
  the model does not count seconds, but it understands "the rest".
- B negates speech **three times over** ("silent", "with her mouth closed", "no
  speech, no words"); A negates once ("says nothing at all"). Redundancy is what
  holds the mouth shut.
- B puts the speech at the very start with "immediately as the shot opens";
  A hedges it inside a four-second window.

## The recipe

```
(English look and framing — head-and-shoulders, or you get burned-in captions.
 She says 「<short Mandarin line>」 immediately as the shot opens. Then she is
 silent for the rest of the shot, <what she does>, with her mouth closed, no
 speech, no words.)
```

- **Front-load the line** if you want a long reaction beat. Put the quiet beat
  first only if you want a quiet *opening*, and then accept that the tail will
  be about one second whatever you ask for.
- **Keep the line under ~45% of the clip.** At 4.5 characters a second that is
  about 20 characters for a 10 s clip. Over that and the head collapses and the
  model writes its own dialogue.
- **Never put stage directions in Mandarin.** 「（轉頭看鏡頭）…（開心地笑）」 was
  read out loud: the take came back saying 「這家真的超愛吃的我覺得它在這家的
  櫻花間…」. Directions go in English, always.
- **Negate speech redundantly** in the quiet beat, and describe the beat's
  length as "the rest of the shot", never in seconds.
- **Grade with ASR, not your ears or an energy meter**, and expect ±0.3 s of
  seed noise on any boundary.

## What is not reachable this way

**Precise seconds.** Nothing in §2 moved when the number moved. If a beat has to
land on a frame, the prompt is the wrong instrument — the two things in H3 that
do carry a time coordinate are the soundtrack under `H3_AUDIO_ANCHOR=1`, which
is exact to the sample, and a frame anchor's RoPE position.

**A silent head and a silent tail at once.** Every take here has one or the
other. The trailing-beat form gives up to 6 s of head and ~1 s of tail; the
front-loaded form gives 0 s of head and up to 6.5 s of tail. No prompt in this
study produced both, and the right-alignment in §3 suggests the model is placing
one block of speech against one end of the clip rather than composing a
timeline.

**Captions, reliably.** They track framing, not wording (see FLAT2V.md,
*Crop to the head, not to the room*). Four of the nineteen takes here carried
burned-in Mandarin, all of them when the beat description pulled the framing
wider — "watching the oyster omelette sizzle" was enough to do it.
