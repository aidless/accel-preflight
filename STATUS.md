# Status

**WIP.** The scanner and the honesty checks run offline. A grade that needs a recomputable DCU capture, or a second first-party GPU, is not met.

## Observed

One private session, not committed in this repository:

- Machine: AMD Radeon PRO W7900, gfx1100, RDNA3
- Stack: ROCm 7.2.1, vLLM 0.16.1.dev+rocm721
- Dates: 2026-10-26/28
- Ledger id: S8

That session is the only `basis=observed` data. It covers flashinfer backends, NEXTN, auto-round, fp8 KV cache, `VLLM_ATTENTION_BACKEND=FLASH_ATTN` failing, and `TRITON_ATTN` recovering. NVFP4 is mentioned in `docs/notes-hygon-ecosystem.md` for the same session, but it was not in the original S8 sentence and the log is absent, so that row is weaker.

Nothing in this tree reruns those commands. `pytest` / `unittest` only checks that the scanner still encodes the ledger.

## Not observed

- Any DCU or HCU boot, install, or failure. Vendor files (S7, S11, S12, S13) are `upstream-doc`.
- Any Instinct / MI300 / MI350 / other RDNA card. `--platform rocm` still applies the W7900 flag blocks to that whole bucket.
- Any non-CUDA accelerator outside the three platform names `nvidia`, `rocm`, `dcu`.
- A fresh re-run of the PyTorch ROCm channel. On 2026-11-09 the install page still showed ROCm 6.3 with a strikethrough. The `+rocm` tag rule was not re-measured.
- Publication. This project is not on PyPI. The CI badge is the GitHub Actions workflow only.

## What 0.4.0 changed without new hardware

String holes that let CUDA-only requirements exit 0 are closed and tested: PEP 503 names, the current FlashInfer distribution names, `whl/cuNNN` index URLs, `+cu` tags on non-torch packages, `+cpu` no longer classified as CUDA, optional-dependency tables, and prefix matches no longer inheriting `ok`. DCU does not inherit S8. See `FAILURE_MODES.md` for the holes that remain.
