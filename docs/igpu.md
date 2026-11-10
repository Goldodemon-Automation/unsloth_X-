# Intel integrated GPUs: the XPU wheel and one pinned card

What an Intel iGPU means for a training run, which PyTorch build it needs, and how to keep a
hybrid machine on the card the user picked. The device registry, the shared-memory budget and the
environment each target sets are covered by [compute-targets.md](compute-targets.md); this page
holds the Intel-specific facts and the checks that separate "the iGPU trains" from "this wheel
cannot".

## What an integrated GPU means for a run

Three facts decide the run, and [compute-targets.md](compute-targets.md) derives all three:

| Fact | For the run | Where it is derived |
| --- | --- | --- |
| The GPU shares system RAM with the OS | the budget is `max(carve-out, 0.6 × RAM)`, not what `torch.xpu` reports | [the integrated-GPU memory budget](compute-targets.md#the-integrated-gpu-memory-budget) |
| No 4-bit training | bitsandbytes has no kernels for an integrated part; the run uses fp16 or bf16 | [what each target can execute](compute-targets.md#what-each-target-can-execute) |
| bf16 is by generation | Intel iGPUs have bf16; AMD gfx10 (RDNA 1/2) advertises it but cannot run it, and takes fp16 instead | [what each target can execute](compute-targets.md#what-each-target-can-execute) |

## The XPU wheel

An Intel iGPU trains only on **PyTorch's XPU build**. The wheel's own metadata says which build is
installed; `torch.version.xpu` is the marker, and a vendor wheel can carry it even when its
version string has no `+xpu` tag:

```bash
python -c "import torch; print(torch.__version__, getattr(torch.version, 'xpu', None))"
```

| The check prints | It means |
| --- | --- |
| a `+xpu` version tag, or a non-`None` `torch.version.xpu` | the XPU build is installed, so the integrated GPU can train |
| `None` for `torch.version.xpu` and no `+xpu` tag | a CPU, CUDA or ROCm build: PyTorch here cannot train the Intel GPU |

When the picker shows `torch_missing`, a reinstall is the fix. The Studio's installer pins the
Python stack — both installers do — and records the wheel it chose in the environment's install
manifest, so reinstalling through it restores the build this machine needs. This page gives no pip
command on purpose: the pinned stack is the installer's decision, not a URL to retype.

For code rather than a shell, `xpu_torch_status()` in `studio/backend/utils/hardware/xpu.py`
returns the same verdict — the build, the device count and the one sentence to show.

## Pinning one GPU on a hybrid machine

A machine with an Intel dGPU and an iGPU enumerates both, and without a pin a run lands on the
first device the runtime happens to list. Pin the card through the oneAPI device selector, by its
Level-Zero ordinal `N`:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:N
```

Two caveats, both about the number:

- **Set it before torch imports.** torch, SYCL and the allocator read it while loading; setting it
  later in the process changes nothing. It goes in the environment the training process starts
  with, the same place
  [the per-target environment](compute-targets.md#environment-a-target-sets) lives.
- **`N` is Level-Zero's enumeration order**, which need not follow PCI order. Once the filter is
  set it re-indexes torch's devices, so the pinned card is what the run then reports as `xpu:0`.

The namespaced `level_zero:N` is deliberate: a bare mask number (`ZE_AFFINITY_MASK`) is ambiguous
as soon as more than one Level-Zero platform is visible. The Studio's `igpu` target sets no device
pin of its own — see
[the environment a target sets](compute-targets.md#environment-a-target-sets) — so this is how a
user names one card. `apply_xpu_selection()` in `studio/backend/utils/hardware/xpu.py` sets the
same value with `setdefault` semantics: a selector the user already set wins, and the function
reports only what it wrote.

## Troubleshooting

Each row is a fact read from the code, not a guess.

| Symptom | Cause | Action |
| --- | --- | --- |
| The picker shows `torch_missing` | the registry's reason reads "PyTorch is not installed in this environment."; on an iGPU host the wheel to install is the XPU build | reinstall through the Studio's installer — a reinstall is the fix, and the installer pins the stack |
| PyTorch is installed, but the check prints `None` for `torch.version.xpu` | the wheel is a CPU, CUDA or ROCm build and carries no XPU runtime | reinstall the XPU build; a CPU wheel on a machine with a GPU is a reinstall, not a GPU that cannot train |
| The XPU build is installed but the GPU still does not appear (device count 0) | the wheel is right and the environment is not: the runtime refuses to initialise — an old driver, a runtime that will not load | fix the driver; reinstalling the wheel will not help |
| A hybrid machine trains on the Intel dGPU instead of the iGPU | nothing pinned the Level-Zero device, and the run landed on the first one enumerated | set `ONEAPI_DEVICE_SELECTOR=level_zero:N` in the environment before torch imports |
| The run refuses a model that fits on paper, or runs out of memory with an empty card | the iGPU shares system RAM; the carve-out torch reports is a floor, not the ceiling | check the planned figure — [the budget section](compute-targets.md#the-integrated-gpu-memory-budget) has the command |
| The picker offers no 4-bit for the iGPU | integrated GPUs do not support 4-bit training | take the 16-bit recommendation — [what each target can execute](compute-targets.md#what-each-target-can-execute) |
