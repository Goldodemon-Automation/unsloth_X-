# Compute targets: CPU, integrated GPU and NPU training

Which device a training run uses, what that device can actually execute, and how much memory it may
plan against. This is the reference for the four targets the Studio can pick between, and for the
limits it states rather than hides.

| Target | What it is | Training status |
| --- | --- | --- |
| `cpu` | The processor, no accelerator | **Supported, slow.** The CoRA/QLoRA recipe on stock `transformers` + `peft`, tuned by `core/training/cpu_tuning.py` (threads, SDPA, dataloader workers, checkpointing, dtype); its budget is *free* RAM, not the installed total. |
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
| `igpu` (Intel) | `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`, `SYCL_CACHE_PERSISTENT=1`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `UNSLOTH_SKIP_PIN_MEMORY=1`, `UNSLOTH_IGPU=1`, plus the three compile switches. No device pin: the card is chosen through `CUDA_VISIBLE_DEVICES`/`ZE_AFFINITY_MASK`, and Unsloth's `igpu` pin raises when no integrated backend answers, which would kill a run whose child lost the GPU |
| `igpu` (AMD) | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, `UNSLOTH_SKIP_PIN_MEMORY=1`, `UNSLOTH_IGPU=1`, plus the three compile switches. No device pin, for the same reason |
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

## How a CPU run is tuned

`core/training/cpu_tuning.py` decides the load shape from host measurements, and `_load_cpu_model`
applies it. The rules are pure functions of what was measured, so they are pinned by tests rather
than discovered on a two-day run:

| Choice | Rule | Why |
| --- | --- | --- |
| `torch.set_num_threads` | one per physical core, approximated as half the usable CPUs when there are more than two | SMT siblings add little to a GEMM and cost cache contention; no portable physical-core count exists |
| thread count under memory pressure | halves when the fp32 weights exceed 60% of available RAM | each thread keeps its own buffers, and those buffers decide between a run and an OOM kill |
| `set_num_interop_threads` | always 1 | torch sizes intra-op and inter-op pools independently, so two core-sized pools multiply into oversubscription, and a CPU run has one graph to schedule |
| attention | `sdpa` when the installed stack has it, else `eager` | the fused kernel instead of materialising the full attention matrix; a model that rejects the kwarg is retried without it rather than failed |
| dataloader workers | 0 on Windows and macOS, `min(4, cores // 4)` on Linux | spawned workers re-import the training module; fork inherits `sys.path`, and a CPU step cannot fill the gap itself |
| gradient checkpointing | on when the fp32 weights exceed 40% of available RAM | recomputing is faster than swapping |
| dtype | fp32 unless the weights cannot fit, then bf16 | the same `cpu_fp32_weights_fit` rule the loader used before, so the plan and the loader cannot drift |

Every unknown measurement degrades to the conservative answer (one thread, no workers, eager, fp32)
instead of raising, and the loader logs the whole plan as one line so a user can see what their
machine chose.

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

## When something does not work

Each entry names the check to run before changing anything, because these are the four cases where
the answer is usually a fact about the machine rather than a bug.

### "My integrated GPU is not in the list"

Check what the driver reports and what this build can open:

```bash
cd studio/backend
python -c "from core.training.device_targets import TargetProbe, list_training_targets; \
p = TargetProbe.detect(); print(p.torch_device); [print(t.id, t.device, t.supported, t.reasons) for t in list_training_targets(p)]"
```

- `torch_device: None` with a GPU present is `torch_missing` on the row: the installed PyTorch has no
  build for that backend. Intel parts need the XPU build, AMD parts the ROCm build; a CPU wheel on a
  machine with a GPU is a reinstall, not a GPU that cannot train.
- The registry lists a device as `integrated` from its **name** (there is no portable API for it).
  A name it does not recognise is treated as discrete, which only changes the memory budget: a
  discrete budget is the card's own, an integrated one is the shared pool.
- Apple Silicon appears as an integrated GPU. If MLX is present, `UNSLOTH_TRAINING_DEVICE=cpu` is the
  only way to train on the CPU there, and it is honoured ahead of MLX.

### "Training on my integrated GPU runs out of memory with an empty card"

The budget is `max(carve-out, 0.6 × RAM)`, not the figure `total_memory` reports, and it is *shared*
with the operating system. Check what was planned:

```bash
python -c "from utils.hardware import accelerator_budget as b; print(b.budget_report(integrated=True, dedicated_bytes=536870912, system_memory_bytes=68719476736))"
```

If `usable_bytes` looks right and the run still fails, the pressure is coming from elsewhere: a chat
model still loaded, a browser, or the dataset pipeline's workers. `UNSLOTH_SKIP_PIN_MEMORY` is set for
this target already, so pinned host memory is not the cause.

### "The NPU is detected but training will not use it"

That is the design, not a fault: no vendor exposes a backward pass to PyTorch. `GET
/api/compute/npu/assist` says what it can do instead and why it refused:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8888/api/compute/npu/assist?requested=assist" | head -40
```

Detection is also gated: without `UNSLOTH_STUDIO_ENABLE_NPU=1` the capability reports
`reasons: ["disabled"]` and the picker row explains it. If the NPU is present and enabled but reports
`no_runtime`, no installed runtime (`openvino`, ONNX Runtime QNN, Vitis AI, FastFlowLM, Lemonade,
Windows ML) can drive that vendor yet — `npu_runtime_providers()` lists all six with what each can
serve.

### "The capacity check says a different device than the picker"

The check resolves the target itself when the request does not name one, in this order: the request's
`compute_target`, then the stored choice, then `auto`. A target's own ceiling outranks a
`gpu_memory_gb` the page sent, because that figure is what torch reported — the carve-out on an
integrated part. `budget_source` in the response says which of the four paths was used
(`target:cpu|igpu|dgpu|npu`, `explicit`, `probe`, `unresolved`).
