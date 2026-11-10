# Compute targets: CPU, integrated GPU and NPU training

Which device a training run uses, what that device can actually execute, and how much memory it may
plan against. This is the reference for the four targets the Studio can pick between, and for the
limits it states rather than hides.

| Target | What it is | Training status |
| --- | --- | --- |
| `cpu` | The processor, no accelerator | **Supported, slow.** The CoRA/QLoRA recipe on stock `transformers` + `peft`; no measurement is quoted, only the order of magnitude. |
| `igpu` | Intel Arc/Xe, AMD APU (RDNA2+), Apple Silicon | **Supported, memory-bound.** Same kernels as a discrete card; the budget is shared with the OS. |
| `dgpu` | NVIDIA, AMD Radeon/Instinct, Intel Arc Pro | **Supported.** The normal case, unchanged. |
| `npu` | AMD XDNA1/2, Intel NPU 3/4, Qualcomm Hexagon, Apple ANE | **Not trainable.** Exposed honestly, usable as an assist device. |

## Where the decision lives

`studio/backend/core/training/device_targets.py` enumerates the candidates. One probe
(`TargetProbe.detect()`) collects measured facts; every capability is a pure function of those facts,
so the rules are testable without hardware.

The order is fixed: **CPU, integrated GPU, discrete GPU, NPU**. The automatic choice is a discrete
GPU, then an integrated one, then the CPU. The NPU is never chosen automatically, because it cannot
run a backward pass and an automatic run would start by refusing itself.

A stored choice is remembered in the app settings (`training_compute_target`), not in the browser, so
a job started from the CLI, another tab or a second machine on the same server runs where the user
chose. `auto` stays `auto`: a laptop that later gains an eGPU must not keep training on the CPU it
resolved to once.

## The integrated-GPU memory budget

This is the one number that decides whether an integrated-GPU run fits, and it is not the number the
driver reports.

`torch.xpu`/`torch.cuda` return the **carve-out** for an integrated GPU — the slice reserved in the
BIOS or by the driver, often 512 MB. The driver then allocates well past it out of system RAM, so the
carve-out is a floor rather than a ceiling. Unsloth plans against the larger of that reservation and
the RAM the run can take without starving the machine:

```
usable = max(carveout, system_memory * 0.6)
```

`0.6` is the fraction in `utils/hardware/accelerator_budget.py` (`SHARED_MEMORY_FRACTION`). Half is
what the Intel driver advertises as its shared ceiling on Windows and AMD's GTT reports something
similar; the extra tenth is headroom for the dataset pipeline and the UI. Below 0.5 the number stops
describing what the driver allows; above 0.7 a background allocation can commit the machine into swap
mid-run, which is worse than refusing at fit time.

The fit check (`core/training/fit.py`) uses it whenever the request names a `compute_target` — the
target's own ceiling outranks a `gpu_memory_gb` the page sent, because the page's figure is the one
torch reported, and that is the carve-out on an integrated part — and reports where the ceiling came
from (`budget_source`) plus the sentence the UI shows (`target_notes`) — "This GPU shares system memory: 38.4 GB of the machine's 64.0 GB is usable for
training, 60% of it." A bare `0.5 GB` beside a 3 GB model reads as a refusal on a 16 GB laptop unless
it also says the card shares that RAM.

## What each target can execute

| | `bf16` | `fp16` | 4-bit (bnb) | Triton | `torch.compile` | FlashAttention |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | — | — | — | — | — | — |
| iGPU (Intel) | yes | yes | — | if installed | — | — |
| iGPU (AMD gfx11+) | yes | yes | — | if installed | — | — |
| iGPU (AMD gfx10) | — (see below) | yes | — | if installed | — | — |
| iGPU (Apple) | — | yes | — | — | — | — |
| dGPU | yes | yes | if installed | if installed | yes | NVIDIA only |
| NPU | — | — | — | — | — | — |

Two rows carry a reason worth writing down.

**No 4-bit on an integrated GPU.** bitsandbytes has no kernels for an integrated part; a 4-bit load
there dequantizes on the CPU every step, which is slower than fp16 and consumes the memory it was
meant to save. The picker shows the recommendation, not the option.

**AMD gfx10 has no usable bf16.** RDNA 1 and RDNA 2 advertise bf16 they cannot run, and Triton's dot
then takes the process down inside LLVM with no Python exception (issue #7922). The registry reports
`bf16: false` and recommends fp16 there. Anything unmeasured is `false` — a `false` costs a slower
path, a wrong `true` kills the run mid-step.

## Environment a target sets

`core/training/compute_target_env.py` runs first in the training subprocess, before any import,
because these values are read at import time by torch, SYCL, the allocator and Unsloth's device
module. Setting them later changes nothing.

| Target | Sets |
| --- | --- |
| `cpu` | `UNSLOTH_TRAINING_DEVICE=cpu`, `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1`, `UNSLOTH_COMPILE_DISABLE=1` |
| `igpu` (Intel) | `UNSLOTH_TRAINING_DEVICE=igpu`, `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`, `SYCL_CACHE_PERSISTENT=1`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `UNSLOTH_SKIP_PIN_MEMORY=1`, `UNSLOTH_IGPU=1`, plus the three compile switches |
| `igpu` (AMD) | `UNSLOTH_TRAINING_DEVICE=igpu`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, `UNSLOTH_SKIP_PIN_MEMORY=1`, `UNSLOTH_IGPU=1`, plus the three compile switches |
| `npu` | `UNSLOTH_TRAINING_DEVICE=npu`, `UNSLOTH_NPU_DEVICE=1`, the three compile switches |
| `dgpu` | nothing: the stock allocator split reuses freed blocks sooner, and a forced `cuda` would break an AMD host where the same card reports as `hip` |

Every value is applied with `setdefault` semantics: a value already in the environment wins, because
a user who typed one has a reason. The module returns only what it actually set, so a run's setup line
cannot claim to have applied somebody else's setting.

`UNSLOTH_TRAINING_DEVICE` is the library-level override and it is **opt-in on purpose**. Unsloth's
kernels assume an accelerator, and a silent CPU fallback would turn a broken driver into a run that is
50-100x slower (an estimate, not a measurement) instead of a run that refuses. `UNSLOTH_TRAINING_DEVICE=igpu` resolves to whichever
accelerator backend is usable (`xpu`, `cuda`/`hip`, `mps`) and refuses rather than quietly training on
the CPU when none answers.

## The NPU

Detection is vendor-neutral (`utils/hardware/npu.py`), and the runtime matrix
(`utils/hardware/npu_providers.py`) lists every provider the runtimes may come from: OpenVINO, ONNX
Runtime QNN, Vitis AI, FastFlowLM, Lemonade, Windows ML. All rows are always present; the installed
ones are marked.

Detection is off unless `UNSLOTH_STUDIO_ENABLE_NPU=1` is set, matching the existing NPU inference
gate.

**No NPU can train a model.** No vendor exposes a backward pass to PyTorch. What the NPU can do is
the work around a training step, and `core/training/npu_assist.py` plans exactly that:

| Task | What the NPU runs |
| --- | --- |
| `tokenize` | Reading and tokenizing the dataset |
| `eval` | Scoring checkpoints while training runs |
| `qat_calibrate` | Calibrating quantized weights |

The backward pass stays on the primary compute device. The plan says so in a sentence, and
`requested="full"` is answered with `refused_reason="no_backward_pass"` rather than being silently
downgraded — a user who asks for NPU-only training deserves the fact, not a quieter run.

The assist queue is bounded, single-consumer and lossy: a full queue drops the submission and the
training step keeps the time it had. A training step that waits on a queue is a slower training step.

## API

| Route | Returns |
| --- | --- |
| `GET /api/compute/targets` | Every device with its capability and recommendation, plus `default`, `requested`, `resolved`, `budget` and the NPU's capability. |
| `POST /api/compute/target` | `{"target": "igpu"}` — validates, remembers, returns what it resolves to. `400` with the reason when it cannot. |
| `GET /api/compute/npu/assist` | The assist plan preview: `enabled`, `mode`, `tasks`, `providers`, `refused_reason`, `notes`. |

`GET /api/compute/targets` includes unsupported devices on purpose. "This machine has an NPU it
cannot train on" is different information from "this machine has no NPU", and hiding the first leaves
a user unable to find out why NPU training is unavailable to them.

## Tests

```bash
cd studio/backend
python -m pytest tests/test_device_targets.py tests/test_accelerator_budget.py \
  tests/test_compute_target_env.py tests/test_compute_target_settings.py \
  tests/test_compute_route.py tests/test_npu_assist.py tests/test_npu_vendors.py \
  tests/test_npu_providers.py tests/test_training_fit.py -q
```

From the repository root: `tests/test_forced_training_device.py` covers the library-level override
with a stand-in torch, including the property that a missing accelerator still raises without it.
