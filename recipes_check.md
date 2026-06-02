# Things to verify in `nsys-multi-report.ipynb` before the workshop

The new "Reading Multiple Traces" notebook (Lab 3, item 3) was written without being able to actually run the commands against the brev container.  A handful of details need a sanity check during a dry run.  All four issues are concentrated in the same notebook — `workspace/jupyter_notebook/nsys-multi-report.ipynb`.

## 1. Recipe names

The notebook hardcodes two specific recipe names:

- `nvtx_sum` — single-report recipe for NVTX range time aggregation.
- `nccl_gpu_overlap_trace` — multi-report recipe for NCCL/compute overlap analysis.

These match the entry in `TODO.md` and the common nsys recipe inventory, but exact names can vary by Nsight Systems release.  Confirm what your container actually ships with:

```bash
nsys recipe --help
```

If either name differs, the closest substitute in the listing is usually obvious — anything containing `nvtx` and `sum` for the former, `nccl` and `overlap` for the latter — and you can swap the names in the notebook cells.

## 2. CLI flag syntax for recipes

The notebook uses `--input <file>` repeated for the multi-report recipe:

```bash
nsys recipe nccl_gpu_overlap_trace \
    --input /workspace/reports/firstOptim_rank0.nsys-rep \
    --input /workspace/reports/firstOptim_rank1.nsys-rep \
    --input /workspace/reports/firstOptim_rank2.nsys-rep \
    --input /workspace/reports/firstOptim_rank3.nsys-rep \
    --output /workspace/reports/overlap_analysis
```

Different nsys versions accept different forms:

- `--input <file>` repeated (what the notebook assumes)
- `--inputs <file1> <file2> ...`
- positional `<file1> <file2> ...`

Confirm with `nsys recipe <recipe-name> --help` and adjust the cells if the flag form differs.

## 3. Wrapper script + torchrun pattern

The notebook writes a small bash wrapper at `/workspace/scripts/nsys_wrap.sh` that contains:

```bash
#!/bin/bash
exec nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --output=/workspace/reports/firstOptim_rank${LOCAL_RANK} \
  --force-overwrite=true \
  python "$@"
```

…and then passes that wrapper to `torchrun` as the entrypoint:

```bash
cd /workspace/source_code && torchrun --nproc_per_node=4 --nnodes=1 --standalone \
    --master_addr="localhost" --master_port=1234 \
    /workspace/scripts/nsys_wrap.sh ddp_optimize.py
```

`torchrun` is documented to accept arbitrary executables and to set `LOCAL_RANK` in each spawned process's environment, so this should produce four `.nsys-rep` files:

```
/workspace/reports/firstOptim_rank0.nsys-rep
/workspace/reports/firstOptim_rank1.nsys-rep
/workspace/reports/firstOptim_rank2.nsys-rep
/workspace/reports/firstOptim_rank3.nsys-rep
```

**Failure mode**: if `LOCAL_RANK` is unset (or empty) in the wrapper's environment, all four ranks will write to `firstOptim_rank.nsys-rep` and only one file (the last writer) will end up in `/workspace/reports/`.  An `ls firstOptim_rank*.nsys-rep` after the cell runs is enough to catch this.

## 4. `--capture-range=cudaProfilerApi` in the wrapper

The wrapper uses `--capture-range=cudaProfilerApi`, which scopes the capture to between `cudaProfilerStart()` / `cudaProfilerStop()` calls in the script.  `ddp_optimize.py` (which the wrapper invokes) already has these calls — `nsys-application.ipynb` uses the same flag against the same script — so this *should* work.

**Failure mode**: if you see "no events captured" warnings or zero-byte `.nsys-rep` files, drop `--capture-range=cudaProfilerApi` from the wrapper and re-run.  Captures will be larger (all epochs included rather than just the second) but the recipe demos will still produce meaningful output.

---

If anything else surfaces during the dry-run, jot it here so the next pass can fix it.
