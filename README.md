# accel-preflight

[![CI](https://github.com/aidless/accel-preflight/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/accel-preflight/actions/workflows/ci.yml)

**Status: WIP.** See [STATUS.md](STATUS.md). No DCU capture is in this repo. There is no PyPI release.

<!-- problem:start -->
A CUDA vLLM or SGLang command and its requirements often die on another accelerator, and the useful error is buried in a child process.
accel-preflight is a zero-dependency static check of flags, package names, and wheel channels against a ledger.
Exit 0 means that ledger found no blocker. It does not mean the target machine booted the model.
<!-- problem:end -->

## OBSERVED

One private session, id S8. The log is not committed, so these rows are not recomputable from this tree.

| What | Scope |
|---|---|
| Machine | AMD Radeon PRO W7900, gfx1100, RDNA3, ROCm 7.2.1, vLLM 0.16.1.dev+rocm721, 2026-10-26/28 |
| Failed there | flashinfer backends, `--speculative-algorithm NEXTN`, `--quantization auto-round`, fp8 KV cache, `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| Recovered there | `VLLM_ATTENTION_BACKEND=TRITON_ATTN` |

`basis=observed` is printed only for that session. `--platform rocm` still applies those flag blocks to the whole `rocm` bucket, which can false-red Instinct. The finding's `scope` line says so.

## UNKNOWN

Anything whose `basis` is not `observed` was not run for this repo.

- **DCU / HCU runtime.** Vendor READMEs (S7, S11, S12, S13) can yield `ok` or `partial` with `basis=upstream-doc`. That is a document, not a boot. `vllm-plugin-das` `ok` means the vendor repo describes the plugin.
- **CUDA-only artifacts on DCU** (`flashinfer-python`, `whl/cuNNN`, `+cuN`) are `blocked` with `basis=upstream-doc`. The refuse is "this string is a CUDA channel," not "a DCU failed in our hands." S8 is not cited for `dcu`.
- **fp8 KV, NEXTN, auto-round, `FLASH_ATTN` on DCU** stay `unknown`. Copying the W7900 result across would be the extrapolation this tool refuses.
- **Other ROCm GPUs** (MI300, MI350, other RDNA, other ROCm versions) were not measured. `+rocm` on torch is `ok` only because the tag says rocm. The PyTorch install page re-read on 2026-11-09 shows ROCm 6.3 struck through.
- **Other non-CUDA accelerators** (Ascend, Moore Threads, Iluvatar, MetaX, Apple GPU, Intel GPU) have no platform key.
- **Unlisted packages** are `PKG-UNKNOWN-001`. Unknown does not fail CI.
- **NVFP4 on NVIDIA.** No separate machine record. The finding is `unknown`. The ROCm NVFP4 row is the notes file, weaker than the other S8 rows, because the original ledger sentence omitted it and the log is absent.

The Chinese evidence ledger is [docs/sources.md](docs/sources.md). Closed and open holes are in [FAILURE_MODES.md](FAILURE_MODES.md).

## Offline

No install and no network. From a clone:

```bash
python3 -m accel_preflight scan \
  --cmd "vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8_e4m3" \
  --platform rocm
python3 -m accel_preflight --prove-red
```

Citations live in `accel_preflight/db.py`. The process does not fetch them. `python3 -m unittest discover -s tests -v` is the same command CI runs, and it does not `pip install`.

Real output of the scan above (exit 1):

```text
accel-preflight · platform=rocm

BLOCKER (1):
  [FLAG-ROCM-005] --kv-cache-dtype fp8_e4m3  →  blocked  basis=observed  (证据 S8)
      fp8 KV cache 在 RDNA3 上不完整（一手实测）
      scope: 仅此一台 gfx1100 / RDNA3 / ROCm 7.2.1 / vLLM 0.16.1.dev+rocm721。不是 MI300/MI350，不是其他 ROCm 版本，不是 DCU。原始日志不在仓库里，不能从本树复算。
      建议：去掉 fp8 KV cache 回到默认精度；显存吃紧时先复核容量预算再谈量化，别拿未验证的 fp8 换显存

summary: 1 blocker, 0 notice
```

Requirements and pyproject (including `[project.optional-dependencies]` and `[dependency-groups]`):

```bash
python3 -m accel_preflight scan --requirements requirements.txt --platform rocm
python3 -m accel_preflight scan --pyproject pyproject.toml --platform dcu --format json
python3 -m accel_preflight --explain S8
```

| Exit | Meaning |
|---|---|
| 0 | No `blocked` row. Not a successful boot. |
| 1 | At least one `blocked` row. |
| 2 | Usage error, including an unknown evidence id. |

## Verification

CI Python: 3.10, 3.11, 3.12, 3.13.

`pyproject.toml` says `requires-python >= 3.8`. CPython 3.8 and 3.9 are not executed in CI.

Discovered tests: 94.

CI command, and the local command, is `python -m unittest discover -s tests -v`, then `python -m accel_preflight --prove-red`. A drift test fails if this README, that command, or the version list diverges.

## License

MIT © 2026 aidless
