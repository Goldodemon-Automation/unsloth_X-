# <img src="images/logo-x-minus-icon.svg" width="46" align="middle" alt="" /> Unsloth X8

A fork of [Unsloth](https://github.com/unslothai/unsloth). It keeps the training stack and pushes on the
Studio: the Scaler can grow **or** shrink a checkpoint, Labs is working engines rather than a preview,
and a run can train on a CPU, an integrated GPU or a discrete one.

> **Branch:** `x8-main`, where this fork is developed. `main` mirrors the imported upstream tree.
> **Releases:** [v1.0.0 … v1.4.1](https://github.com/Goldodemon-Automation/unsloth_X-/releases)

---

## Branding: this fork is blue, upstream is green

Blue is this fork, green is upstream Unsloth. The mark is
[`images/logo-x-minus-icon.svg`](images/logo-x-minus-icon.svg) — X in blue, dash in a lighter blue, on a
black plate. Alternates and the palette are in [images/logo-options.html](images/logo-options.html).

---

## Install

Signed Windows installers are attached to every release since `v1.4.0`:

| Asset | Use it when |
| --- | --- |
| `…_x64-cpu-gpu-setup.exe` | **Default.** Probes the machine and installs what it can run |
| `…_x64-gpu-setup.exe` | You want the GPU build; it auto-detects CUDA or Vulkan |
| `…_x64-cpu-setup.exe` | No GPU, or no CUDA download |
| `…_x64-cpu-npu-setup.exe` | The CPU stack with `UNSLOTH_STUDIO_ENABLE_NPU=1` already set |

An `-npu` installer changes one thing: the gate. The NPU is driven by Lemonade and FastFlowLM, which the
Studio installs on first use because they exist for AMD XDNA 2 only — so it is the same Python stack,
launched with the NPU enabled. There is no GPU + NPU installer: the hardware with an NPU today is a
Ryzen AI APU whose GPU shares the same system memory, so the pairing would advertise two pools where
the machine has one.

**NPU limits no installer changes:** inference and assist only. Tokenizing, checkpoint scoring and QAT
calibration can run there; training cannot, because no vendor exposes a backward pass.

Linux and macOS build from source, below.

---

## Run it

### Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Goldodemon-Automation/unsloth_X-/blob/x8-main/studio/Unsloth_Studio_X8_Colab.ipynb)

### Local

```bash
git clone -b x8-main https://github.com/Goldodemon-Automation/unsloth_X-.git
cd unsloth_X
bash studio/setup.sh --local     # installs this checkout editable, plus prebuilt engines
unsloth studio -p 8888           # http://127.0.0.1:8888
```

`setup.sh --local` is what makes the checkout win over a published `unsloth` wheel, so run it from the
checkout rather than from a copied folder. `UNSLOTH_SKIP_WHISPER_INSTALL=1` and
`UNSLOTH_SKIP_AUDIO_CPP_INSTALL=1` skip the speech and audio engines.

### Development

Two terminals, backend first:

```bash
python studio/backend/run.py --port 8888
cd studio/frontend && npm install && npm run dev   # http://127.0.0.1:5173, proxies /api to 8888
```

The desktop shell (`studio/src-tauri/`) hosts the same UI; the Vite server and the desktop app talk to
one backend.

---

## What this fork adds

| Area | What it does |
| --- | --- |
| **Scaler** | Grow a checkpoint (more depth, wider feed-forward) or shrink it, planned from `config.json` alone so the slider answers before weights load |
| **Fit check** | Sizes a run from the card before anything starts, so an out-of-memory job is caught at plan time |
| **Failures explained** | A failed run names the likely cause and the setting to change, with the raw error kept one click away |
| **Devices** | CPU, integrated GPU, discrete GPU and NPU in one registry, with honest capability per device and a picker in the Train wizard |
| **Optional host allow-list** | `UNSLOTH_TRUSTED_HOSTS` refuses a foreign `Host` when you set it, and changes nothing when you do not |
| **Caches that earn it** | Device probes, NPU runtime scans, Hub metadata and dataset fingerprints are cached on a key that changes when the answer would — bounded, self-expiring, never a different answer |
| **Labs** | CMT, Scale-before-training, Benchmarks, Artifacts, Pipelines, the shrink repair preset and Cleanup — each an engine with tests, behind one switch |
| **Automated training** | Samples 2-8 datasets, runs 2-72 trainings across CMT/CPT/FFT, benchmarks each, keeps the best and continues over rounds |
| **Run records** | Every merge, scale, CMT and pipeline writes `run_manifest.json` and the license notice beside the checkpoint |
| **Licensing** | Original additions are [NRC-3](LICENSE-NRC-3.md); inherited code keeps its upstream license |

### The Scaler, in two sentences each

**Grow** is bit-exact at step 0 — a new block contributes exactly zero until training unlocks it, so the
checkpoint computes the base function while it waits. Width growth lands within ~1e-5 of the base logits
rather than exactly, because float32 re-associates the widened sum.

**Shrink** cannot be function-preserving: a removed block is a removed residual, a narrowed MLP is a
smaller projection. The run reports the drift it measured, and the Labs **shrink repair** preset is the
recovery pass — measure, train, measure again, pass or fail against a tolerance you set. If you only
need training to fit in less VRAM, QLoRA or a smaller base is less work.

### Runs on what you have

| Target | Training |
| --- | --- |
| CPU | Yes, slow: CoRA/QLoRA on stock `transformers` + `peft`, tuned for the machine and planned against free RAM |
| Integrated GPU | Yes: Intel Arc/Xe, AMD APUs, Apple Silicon — budgeted from memory shared with the system |
| Discrete GPU | Yes; the normal case |
| NPU | No, by hardware: it hosts inference and plans as an assist device |

The reference — the registry, the shared-memory budget, the environment each target sets and why — is
[docs/compute-targets.md](docs/compute-targets.md).

---

## Repository map

| Path | What it is |
| --- | --- |
| `unsloth/` | Unsloth Core: kernels, loaders, patchers, `models/scaling.py` |
| `studio/backend/` | FastAPI server: training worker, merges, Scaler, export, Labs, hardware probe |
| `studio/frontend/` | React + Vite + Tailwind UI, TypeScript |
| `studio/src-tauri/` | Desktop shell hosting the Studio |
| `docs/` | Repository docs |

The full map, including the sidebar's surfaces, is [docs/README.md](docs/README.md). Merging, scaling and
export are engines under the run layer (jobs, resource arbitration, `run_manifest.json`), not separate
applications.

---

## Checks

```bash
cd studio/frontend
npm run typecheck    # tsc over src and tests
npm test             # node --test over tests/**/*.test.ts
npm run i18n:check   # locale overlay parity
npm run biome:check  # lint + format

cd studio/backend
python -m pytest tests -q
```

From the repository root, `pre-commit run --all-files` runs Ruff and the repo's lint scripts. The Scaler
and device suites are named where they live: `test_advanced_scaling*.py`, `test_engine_shrink.py`,
`test_device_targets.py`, `test_accelerator_budget.py`, `test_compute_target_env.py`,
`test_compute_route.py` and `test_npu_assist.py`.

`?perf=1` in the Studio's URL turns on the instrumentation layer; `reportLines()` then lists the slowest
measurements, including one span per route mount.

A suite that looks mysteriously red on Windows is usually line endings: `.gitattributes` asks for LF
under `studio/frontend/`, `core.autocrlf=true` checks out CRLF, and several tests pin exact source text.

---

## Docs

| Doc | Covers |
| --- | --- |
| [docs/labs.md](docs/labs.md) | The Labs switch, the experiment registry, Artifacts and Pipelines |
| [docs/simulated-training.md](docs/simulated-training.md) | The Debugging simulator: a job with no hardware behind it |
| [docs/compute-targets.md](docs/compute-targets.md) | CPU, integrated-GPU and NPU training, the budget, and troubleshooting |
| [docs/igpu.md](docs/igpu.md) | Intel iGPUs: the XPU wheel, pinning one GPU, symptom-first fixes |
| [docs/ui-performance.md](docs/ui-performance.md) | The perf layer: what it measures and how to read it |
| [docs/security.md](docs/security.md) | Threat model, the controls that exist, and the three open items |
| [docs/theming.md](docs/theming.md) | Palettes and accent tokens |
| [docs/README.md](docs/README.md) | Repository map and the local CRLF trap |

---

## Releases

Tags are on `x8-main` and independent of the wheel version in `unsloth/_version.py`, which tracks
upstream. Notes for each are in [docs/releases/](docs/releases/).

| Tag | Theme |
| --- | --- |
| [`v1.0.0`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.0.0) | CPU and integrated-GPU training |
| [`v1.1.0`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.1.0) | NPU detection and NPU-assisted training |
| [`v1.2.0`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.2.0) | Capability card and the perf layer |
| [`v1.2.1`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.2.1) | Windowed run history |
| [`v1.2.2`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.2.2) | The review fixes |
| [`v1.2.3`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.2.3) | Vendor-correct NPU refusals |
| [`v1.3.0`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.3.0) | Tuned CPU runs, tested device decisions |
| [`v1.4.0`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.4.0) | The mark, security hardening, installers |
| [`v1.4.1`](https://github.com/Goldodemon-Automation/unsloth_X-/releases/tag/v1.4.1) | Intel detection, matching backend, opt-in telemetry |

---

## License

Three sets of terms, no overlap:

- **`unsloth/`** — Apache-2.0, inherited from upstream.
- **`studio/`** — AGPL-3.0, see [studio/LICENSE.AGPL-3.0](studio/LICENSE.AGPL-3.0).
- **Original additions** — the Labs engines, the trainer surface, the run records — [NRC-3](LICENSE-NRC-3.md).
  Files carrying the NRC-3 header are covered; the Scaler lives in upstream-licensed files and keeps
  those terms.

NRC-3 allows private, internal and commercial use, modification and sharing with attribution, and
reserves five things that need written permission: relaunching the work as a rebranded product (a
different name, logo or interface, a near-miss name, an "official" claim), running it as a hosted service
for other people, public distribution of packaged builds at scale, folding it into something you sell,
and stripping its notices or relicensing it. Publishing your changed source on a code host stays allowed.
Model weights, tokenizers and datasets keep whatever license came with them.

Produced checkpoints carry `LICENSE-NRC-3.txt` beside `run_manifest.json`, so a folder keeps those terms
wherever it is copied.

---

## Contributing

Issues and pull requests are welcome. Small fixes can go straight to a PR; for anything larger, open an
issue with the shape you have in mind first.

The most self-contained places to start: `studio/backend/core/labs/` (each engine is a module with its
own tests), `studio/frontend/src/features/labs/`, and `studio/backend/core/advanced/` (custom eval tasks,
the repair preset, adapter merging) — each one module, each with tests beside it.
