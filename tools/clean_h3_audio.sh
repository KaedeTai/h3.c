#!/bin/zsh
# Clean up an H3-generated soundtrack.
#
# H3's audio latent is [32,2,T] - genuinely stereo - and the DiT denoises both
# channels from independent noise. They only converge on the same waveform if
# you give them enough steps, so an under-stepped run leaves a decorrelated
# difference between L and R. That difference is pure artifact: the voice
# reference is mono, so there is no real stereo information to protect. It is
# also inaudible to any mono analysis, which is why it hid for so long.
#
# Side-channel level relative to mid, measured on the doctor takes:
#
#   reference wang_line.wav   none (true mono)
#   base 20-step 384x512      -29.2 dB
#   base 30-step 480x640      -25.3 dB
#   base 20-step 480x640      -23.8 dB
#   turbo 4-step              -17.3 dB   <- what you hear as hiss around the voice
#
# Collapsing to mid kills the side channel outright and, because the in-phase
# noise in the two channels is independent too, takes another 3 dB off what is
# left. Nothing of the voice is lost.
#
# Usage: clean_h3_audio.sh IN.mp4 [OUT.mp4]
set -e
IN="$1"; OUT="${2:-${IN:r}_clean.mp4}"
[[ -f "$IN" ]] || { echo "usage: $0 IN.mp4 [OUT.mp4]"; exit 1; }

ffmpeg -y -v error -i "$IN" -c:v copy \
  -af "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=60,afftdn=nr=12:nf=-38:tn=1" \
  -c:a aac -b:a 128k -ac 1 -ar 32000 \
  "$OUT"
echo "wrote $OUT"
