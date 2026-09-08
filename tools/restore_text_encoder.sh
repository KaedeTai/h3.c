#!/bin/zsh
# Put the BF16 text encoder back.
#
# The bundles ship an int4 g32 text encoder (15 GB). The BF16 originals were
# deleted to reclaim disk, which is safe only because they are byte-identical to
# the public release: MiniMaxAI/MiniMax-H3 on Hugging Face, public and ungated,
# FL2VA/text_encoder/model-*-of-00014.safetensors. Verified before deleting --
# all 14 shard sizes matched and shard 14's sha256 was
# e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b on both
# sides.
#
# FL2VA and Ref2VA carried the SAME text encoder files here (one set of inodes,
# hardlinked four ways), so one download serves every bundle.
#
#   restore_text_encoder.sh [full|slim|int4]
#     full  re-download 63 GB and use it as-is
#     slim  re-download, strip layers 50-63 and lm_head, use that   (default)
#     int4  re-download, strip, requantise -- only needed if text_encoder_int4
#           was deleted too
set -e
cd "$(dirname "$0")/.."
MODE=${1:-slim}
if [ ! -d text_encoder_full ]; then
  echo "==> downloading MiniMaxAI/MiniMax-H3 FL2VA/text_encoder (~63 GB)"
  command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
  hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/text_encoder/*' \
     --local-dir /tmp/h3-te-download
  mv /tmp/h3-te-download/FL2VA/text_encoder text_encoder_full
fi
case "$MODE" in
  full) SRC=text_encoder_full ;;
  slim) [ -d text_encoder_slim ] || python3 tools/strip_text_encoder.py \
            text_encoder_full text_encoder_slim
        SRC=text_encoder_slim ;;
  int4) [ -d text_encoder_slim ] || python3 tools/strip_text_encoder.py \
            text_encoder_full text_encoder_slim
        [ -d text_encoder_int4 ] || python3 tools/quantize_text_encoder_int4.py \
            text_encoder_slim text_encoder_int4 --group 32
        SRC=text_encoder_int4 ;;
  *) echo "usage: $0 [full|slim|int4]"; exit 1 ;;
esac
for B in MiniMax-H3 MiniMax-H3-turbo-int8; do
  for V in FL2VA Ref2VA; do
    [ -d "$B/$V" ] || continue
    rm -rf "$B/$V/text_encoder"
    cp -al "$SRC" "$B/$V/text_encoder"
    echo "$SRC -> $B/$V/text_encoder"
  done
done
echo "done; verify with:  ./h3 -d ./MiniMax-H3-turbo-int8 --info"
