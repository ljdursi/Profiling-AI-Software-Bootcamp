#!/usr/bin/env bash
# Execute the workshop notebooks in place, capturing their outputs.
# Designed to be executed INSIDE the docker container at /workspace.
#
# Notebooks are run --inplace so the real 4xL4 outputs are written back into
# the repo copies (next to any H100 "expected output" already in the markdown).
# --allow-errors keeps outputs even when a cell raises, so a single bad cell
# doesn't discard the whole run; failures are detected afterwards by scanning
# the executed notebook for error outputs.
#
# Order follows the Table of Contents in workspace/start_here.ipynb.
# The two Appendix notebooks (appendix-multinode, appendix-te-advanced) are
# reference material and are intentionally NOT run here.
set -u

OUTDIR=/tmp/nbout
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
: > "$SUMMARY"

NOTEBOOKS=(
  # Intro Lab: Introduction to Profiling
  jupyter_notebook/intro-profiling-concepts.ipynb
  jupyter_notebook/intro-pytorch-profiler.ipynb
  jupyter_notebook/intro-profiler-exports.ipynb
  jupyter_notebook/intro-amp.ipynb
  jupyter_notebook/intro-nsys.ipynb
  jupyter_notebook/nsys-gui-walkthrough.ipynb
  # Lab 1: System Topology
  jupyter_notebook/system-topology.ipynb
  # Lab 2: Distributed Training Strategy
  jupyter_notebook/data-parallelism.ipynb
  jupyter_notebook/model-parallelism.ipynb
  # Lab 3: Performance Overview
  jupyter_notebook/nsys-application.ipynb
  jupyter_notebook/nsys-trace.ipynb
  jupyter_notebook/nsys-multi-report.ipynb
  jupyter_notebook/nsight_advanced.ipynb
  # Lab 4: Transformer Engine
  jupyter_notebook/transEng.ipynb
  jupyter_notebook/nsys-fp8.ipynb
)

cd /workspace

for nb in "${NOTEBOOKS[@]}"; do
  name=$(basename "$nb" .ipynb)
  log="$OUTDIR/${name}.log"
  echo "=== START $nb $(date -Iseconds) ===" | tee -a "$log"
  start=$(date +%s)
  # Execute in place. The kernel's cwd is the notebook's own directory, so
  # relative paths (e.g. cd ../source_code) resolve as the notebook expects.
  jupyter nbconvert --to notebook --execute --inplace "$nb" \
      --allow-errors \
      --ExecutePreprocessor.timeout=1800 \
      --ExecutePreprocessor.kernel_name=python3 \
      >>"$log" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$((end-start))

  # --allow-errors makes nbconvert return 0 even when a cell raised, so inspect
  # the executed notebook for error outputs to decide PASS/FAIL.
  cell_errors=$(python3 - "$nb" <<'PY'
import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
n = 0
for c in nb.cells:
    if c.get("cell_type") == "code":
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                n += 1
print(n)
PY
)

  if [ "$rc" -eq 0 ] && [ "${cell_errors:-1}" -eq 0 ]; then
    status=PASS
  else
    status=FAIL
  fi
  printf "%s\t%s\t%ds\trc=%d\tcell_errors=%s\n" \
    "$status" "$nb" "$elapsed" "$rc" "${cell_errors:-?}" | tee -a "$SUMMARY"
  echo "=== END $nb $status rc=$rc cell_errors=${cell_errors:-?} elapsed=${elapsed}s ===" | tee -a "$log"
done

echo "===== SUMMARY ====="
cat "$SUMMARY"
