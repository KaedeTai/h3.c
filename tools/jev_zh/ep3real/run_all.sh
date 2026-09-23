#!/bin/sh
# All four models over the whole set (dev+test in one pass; split afterwards by the items' own field).
cd "$(dirname "$0")/.."
for m in v2 v3 v1; do
  python3 eval_bundle.py runs/$m/bundle ep3real/ep3_all.json ep3real/out_$m.json zh || exit 1
done
python3 eval_bundle.py ~/models/open-jev-deberta-v3-large ep3real/ep3_all.json ep3real/out_en.json zh || exit 1
echo ALLDONE
