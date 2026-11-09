# Failure modes

A green exit means the ledger did not find a `blocked` row. It does not mean the model started, and it does not mean a DCU was measured.

`basis` on every finding is the part that keeps those two apart:

| basis | Meaning |
|---|---|
| `observed` | First-party evidence, and the status is not `unknown`. In this repo that is only S8: one Radeon PRO W7900, gfx1100, ROCm 7.2.1, vLLM 0.16.1.dev+rocm721, 2026-10-26/28. The log is not in the tree. |
| `upstream-doc` | A README or install page. `ok` here means the document lists that channel, not that we booted it. |
| `unknown` | No conclusion. Unknown does not fail CI. |

## False greens that used to let CUDA-only inputs through

These were string bugs. They are closed by tests in `tests/test_honesty.py`, not by a new GPU run.

| Input | What happened | What happens now |
|---|---|---|
| `flash_attn` / `flash.attn` | Not equal to `flash-attn`, so `PKG-UNKNOWN-001`, exit 0 | PEP 503 name fold, then the flash-attn rule (`variant` on ROCm, still not a blocker) |
| `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` | Only the old name `flashinfer` was exact | Exact names from the 2026-11-09 README re-read (S1), `blocked` on `rocm` and `dcu` |
| `--extra-index-url https://download.pytorch.org/whl/cu118` plus `torch==2.9.1` | Index lines were skipped; the pin was `unknown`; exit 0 | `PKG-INDEXURL-001` `blocked` on `rocm` and `dcu` |
| `torch @ https://download.pytorch.org/whl/cu128/torch-…%2Bcu128-….whl` | No `==` local version, so `unknown` | URL/`%2BcuN` is a CUDA channel string, `blocked` on `rocm` and `dcu` |
| `torchvision==0.20.1+cu128` | Only the package name `torch` was inspected | Any non-torch `+cuN` or `whl/cuN` is `blocked` on `rocm` and `dcu` |
| `torch==2.9.1+cpu` on `nvidia` | `cpu`.startswith(`cu`) took the CUDA branch and returned `ok` | Family `cpu` is `unknown` on every platform |
| `vllm-custom-build` | Prefix of `vllm` inherited `rocm=ok` | Prefix of an `ok` rule is downgraded to `partial` |
| `[project.optional-dependencies]` / `[dependency-groups]` | Only the first `dependencies = [...]` array was read | Those two sections are scanned |

`blocked` on `dcu` for the rows above cites S1 or S9 (`basis=upstream-doc`). It means "this artifact is a CUDA channel string." It does not mean a DCU failed in our hands.

## False claims about DCU

| Claim | What is actually true |
|---|---|
| `vllm-plugin-das` / `sglang-das` are `ok` on `dcu` | Vendor READMEs (S13, S12). `basis=upstream-doc`. No DCU capture exists here. |
| `vllm` / `sglang` are `partial` on `dcu` | Upstream or vendor docs say a plugin is required (S11, S7, S12). Not a run. |
| FlashInfer flags are `blocked` on `dcu` | S1, CUDA-only distribution. S8 is not cited for `dcu`. |
| fp8 KV cache, NEXTN, auto-round, `FLASH_ATTN` on `dcu` | No platform key. Status `unknown`. Copying the W7900 result onto DCU is the extrapolation this repo refuses. |
| `torch==…+rocm…` on `dcu` | `unknown`. A ROCm wheel tag is not a DCU result. |
| `torch==…+cu…` on `dcu` | `blocked` with `basis=upstream-doc` (S9 channel mismatch), not an observed DCU crash. |
| NVFP4 on non-Blackwell NVIDIA | Not written. `nvidia` has no key, so the finding is `unknown`. The older notes sentence that said those cards fail has no separate machine record. |

`--platform rocm` is one bucket. S8 `blocked` rows are one gfx1100 card. Applied to Instinct they can false-red. They are not a measurement of MI300/MI350. The finding carries `scope` only when `basis=observed`.

`PKG-SGLANG-001` `rocm=ok` cites the upstream Instinct table (S4), not the W7900 session.

## Holes that are still open

Exit stays 0 for each of these. They are pinned by tests so a later change cannot pretend they are covered.

- `torch==2.9.1` with no local version and no `whl/cu` URL is `unknown`. This ledger does not identify the default PyPI torch build.
- `+cpu` is `unknown`, not `blocked`.
- `-r other.txt` / `--requirement` is not opened. The line becomes `PKG-INCLUDE-001` (`unknown`). CUDA-only lines inside the included file are invisible, and the exit code can stay 0.
- `--quantization auto_round` (underscore) does not match `auto-round`. Flag matching is a substring test (`pat in value`), so it also double-fires when one token contains several patterns, and it can false-red an unrelated value that merely contains `awq` or `fp8`.
- Environment variables are read only when they are prefixes of the scanned command. The ambient process environment is ignored.
- `setup.cfg`, dynamic PEP 621 dependencies, and `tool.uv` source tables are not parsed. A quoted `flashinfer` under `[tool.other]` is ignored on purpose.
- A CUDA wheel index whose path does not contain `whl/cu` plus digits is not recognized.
- Prefix of a `blocked` name stays `blocked`. A future package literally named `flashinfer-rocm` would still be refused.
- Accelerators with no platform key (Ascend, Moore Threads, Iluvatar, MetaX, Apple GPU, Intel GPU) cannot be selected. Unlisted packages are `unknown` and do not fail CI.
- `+rocm` on `rocm` for torch is `ok` because the tag says `rocm`. The 2026-11-09 install page shows ROCm 6.3 struck through, so that `ok` is not "the channel is still published" and not a boot.

## What CI does not prove

CI runs `python -m unittest discover -s tests -v` and `python -m accel_preflight --prove-red` on CPython 3.10–3.13. It does not install a wheel, talk to a GPU, or re-fetch the evidence URLs. `requires-python` is `>=3.8`; 3.8 and 3.9 are not executed.
