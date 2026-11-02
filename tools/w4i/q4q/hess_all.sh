#!/bin/bash
# hess_all.sh : X^T X of ffn_in / ffn_down for ALL 64 layers in one llama.cpp CPU pass over the public calibration set
# (work/calib, 151,552 tokens, ids identical to tools/w4i calib-SHA256SUMS).  Batches of ~24K tokens: dump -> acc -> delete.
# usage: hess_all.sh GGUF WORKDIR (WORKDIR/calib = tools/w4i/calib.py ids; ACTDUMP = the tools/w4i actdump binary)
M=$(readlink -f $1); cd $2
HD=hess; T=$(cd $(dirname $0) && pwd)
mkdir -p $HD
run() {
  rm -rf htmp; mkdir htmp
  W4C_KINDS=ffn_in,ffn_down DUMP_THREADS=30 nice ${ACTDUMP:?set ACTDUMP} --model $M --out htmp "$@" >> hess-dump.log 2>&1 || { echo "dump failed $*"; exit 1; }
  OPENBLAS_NUM_THREADS=32 nice python3 $T/hess_acc32.py htmp $HD >> hess.log 2>&1 || { echo "acc failed"; exit 1; }
  printf '%s\n' "$@" >> $HD/fed
  echo "$(date +%T) += $*"
}
B=(); n=0
for f in calib/train-*.ids; do
  grep -qx "$f" $HD/fed 2>/dev/null && continue
  B+=($f); n=$((n + $(tr ',' '\n' < $f | wc -l)))
  if [ $n -ge 24000 ]; then run "${B[@]}"; B=(); n=0; fi
done
[ ${#B[@]} -gt 0 ] && run "${B[@]}"
rm -rf htmp; touch $HD/DONE; echo HESSDONE
