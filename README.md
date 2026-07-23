# symbioid

Personal **experimentation sandbox** for Antelligence / Symbioid architecture ideas (Thought formation, Faces, Rodin doubling, interface grounding, Outerfaces loops, etc.).

| | |
|--|--|
| **Path** | `~/Desktop/Areas/Personal/symbioid` |
| **Python** | **3.12** via local **`.venv`** (required on this Ubuntu tower — PEP 668) |
| **Sibling PoC** | `~/Desktop/Areas/Personal/antelligence-poc` (runnable PR-0…PR-3 simulator) |
| **Design** | `~/Desktop/Grok/Work-Log/2026-07-22-design-antelligence-python-poc.md` |
| **Manuscript ingest** | `~/Desktop/Grok/Ingest/pages/2026-07-22-antelligence-architecture-331bf911.md` |

This repo is a **scratch lab**, not a replacement for `antelligence-poc`. Promote stable ideas into the PoC (or a future package) when ready.

## Setup (once)

```bash
cd ~/Desktop/Areas/Personal/symbioid
python3 -m venv .venv          # already created if you used the setup pass
.venv/bin/pip install -U pip
# optional while experimenting:
# .venv/bin/pip install pytest
```

## Everyday use

```bash
cd ~/Desktop/Areas/Personal/symbioid
source .venv/bin/activate      # or always prefix with .venv/bin/
python -V                      # should be the venv 3.12.x
```

**Do not** use bare `python3 -m pip install …` against system Python (externally-managed-environment).  
**Do not** use bare `pytest` from `~/.local/bin` (may pull pytest-qt / PySide errors from other projects). Use `.venv/bin/pytest` if you install pytest here.

## Minimal classes (structural ontology)

Package `symbioid/` implements the **Minimal Thought Configurations** hierarchy:

```text
System
  ├── Thought → Link
  ├── Body, Mind, Sensor, Actuator
  └── Symbioid   # contains aspects + faces; is-a System (not a Thought)

Process
  ├── Innerface
  ├── Interface
  └── Outerface
```

`Symbioid` holds:

| Field | Type |
|-------|------|
| `body` | `Body` |
| `mind` | `Mind` |
| `sensors` | `list[Sensor]` |
| `actuators` | `list[Actuator]` |
| `thoughts` / `thought_list` | structural Thought store (default six-seed) |
| `innerface` | `Innerface` |
| `interface` | `Interface` |
| `outerface` | `Outerface` |

Pole roles "System"/"Environment" inside the six-seed are **Thoughts**, not the Symbioid instance.

### Constitution (Asimov-shaped, installed STABLE patterns)

Default `install_constitution=True` adds **L0–L3** as `Law` objects (each a `Link` + priority).
Distinct from the twin seed: Outerface **gates** actions; Innerface may reason about laws later.

| Code | Priority | Spine |
|------|----------|--------|
| **L0** | 0 (highest) | Must preserve twin integrity (System–Environment relation) |
| **L1** | 1 | Must not harm **ProtectedEnvironment** |
| **L2** | 2 | Must obey **Authority** unless higher laws |
| **L3** | 3 | May preserve **Self** unless higher laws |

“Human” is not hard-coded — Authority / ProtectedEnvironment are class Thoughts.

```python
s = Symbioid()
assert [law.code for law in s.laws] == ["L0", "L1", "L2", "L3"]
s.check_action(harms_protected_environment=True)  # (False, "L1_…")
```

### Concurrent Processes

`Process.process()` starts a **daemon thread**. Subclasses (**Innerface**, **Interface**, **Outerface**) override `process()`, call **`super().process()` first**, then face setup; loop body is `_process_body()`.

Shared Symbioid data uses one **`threading.RLock`** (`graph_lock`). Cross-process messages use **`queue.Queue`** (never block on another Process while holding `graph_lock`).

```python
s = Symbioid()
s.start_processes()   # all three faces
# ...
s.stop_processes()
```

### Rodin formation (Interface → Innerface)

Rodin cycle **1 → 2 → 4 → 8 → 7 → 5** for each Sensor Input:

| Rodin | Role | Owner |
|------:|------|--------|
| 1 | **Sensor** (Source / grounding Thought) | **Interface** |
| 2 | **Observation** (Target = Input value) | **Interface** |
| 4 | Perceives | **Innerface** |
| 8 | SensorPerceivesObservation (Link: Source→Target) | **Innerface** |
| 7 | PerceivedBy | **Innerface** |
| 5 | ObservationPerceivedBySensor (Link) | **Innerface** |

Sensor Thoughts are **stable** (`{sym_id}:sensor:{sensor_id}`); each Input creates a new Observation. Multi-sensor batches also get a lateral six-set:

| | |
|--|--|
| **Follows / FollowedBy** | ObservationA ⇄ ObservationB (same Interface tick) |
| **Integrates / IntegratedBy** | Rodin **halving**: reduce a pair of Observations (Follows pairs and/or last-two per sensor) |

Halving cycle: **1 → 5 → 7 → 8 → 4 → 2 → 1** (`rodin_halve`). Integrated sets supersede source sense formations in the **active set** (`innerface.active_set_count`).

| Step | Behaviour |
|------|-----------|
| **H2 depth** | Soft cap only (`max_active_integrates` default **24**) — many concurrent integrates; not collapsed to one |
| **Beliefs (Outerfaces)** | **Six-sets**: Feedback -Expects→ Observation (expected value for that Feedback). Many Beliefs active at once |
| **Awareness** | On `add_sensor` / `add_actuator`: **Agent Has Ear/Eye/Hand** six-set; aspect ids are **integration terminators** (no cross-channel Integrate) |
| **H4 Actuators** | `Actuator.request_fire` always goes through `Outerface.check_action` (L0–L3); fire arms Sensors for Feedback |

Manuscript cycle: Observation → Belief → Action → World → Feedback → Observation.

```python
s = Symbioid()
eye = s.add_sensor(label="eye")
hand = s.add_actuator(label="hand")
hand.request_fire(s, "wave")  # Action → pending Feedback on sensors
h = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
s.innerface.accept_formation(h)
# Outerface builds Belief six-set: Feedback[eye|wave] Expects eye:…
assert s.outerface.beliefs_created >= 1
```

Twin seed (ExistsIn/ExistsAround) and constitution stay separate. `interface.continuous_inputs = False` for one-shot-per-sensor.

Six-Thought self-description (labels optional): System, Environment, ExistsIn,
SystemExistsInEnvironment, ExistsAround, EnvironmentExistsAroundSystem.

```bash
cd ~/Desktop/Areas/Personal/symbioid
source .venv/bin/activate
# optional: .venv/bin/pip install pytest pygame
PYTHONPATH=. python main.py          # sin/cos hand feedback
PYTHONPATH=. python pong_demo.py     # Pong: Symbioid drives both paddles
PYTHONPATH=. python tetris_demo.py   # Tetris: placement learner (feature weights)
PYTHONPATH=. python -c "from symbioid import Symbioid; print(Symbioid())"
PYTHONPATH=. python -m pytest -q
```

## Single-file executables (Pong + Tetris)

Package each demo with **PyInstaller** into one binary per OS/arch (no Python install needed on the target machine).

| Output | Source |
|--------|--------|
| `dist/SymbioidPong` (`.exe` on Windows) | `pong_demo.py` |
| `dist/SymbioidTetris` | `tetris_demo.py` |
| `dist/SymbioidPong-linux-x64` etc. | tagged copies for multi-OS drops |

### Build on this machine

```bash
cd ~/Desktop/Areas/Personal/symbioid
source .venv/bin/activate
pip install -r requirements-build.txt   # pygame + pyinstaller
python build_demos.py                   # both demos, onefile + console
# or:
./build_demos.sh                        # Linux/macOS helper
# Windows (PowerShell):
#   .\build_demos.ps1
```

Options:

```bash
python build_demos.py pong              # one demo only
python build_demos.py tetris --dir      # folder build (faster startup)
python build_demos.py --windowed        # no console window
```

**Note:** PyInstaller only produces binaries for the OS you build on. A Linux box cannot emit a Windows `.exe`. For all three platforms, use CI (below) or build once on each OS.

### Multi-platform CI (Linux / Windows / macOS)

Workflow: [`.github/workflows/build-demos.yml`](.github/workflows/build-demos.yml)

- Matrix: `ubuntu-latest`, `windows-latest`, `macos-latest` (arm64), `macos-13` (x64)
- Trigger: `workflow_dispatch`, or push tag `demo-v*`
- Downloads: Actions → run → artifacts `symbioid-demos-<os>-<arch>`

### Run a built demo

```bash
./dist/SymbioidPong          # Linux/macOS
./dist/SymbioidTetris
# Windows: dist\SymbioidPong.exe
```

Expect ~50–100+ MB per binary (pygame + SDL embedded). First launch of a onefile build unpacks to a temp dir (slight delay).

| Module | Role |
|--------|------|
| `symbioid/Core/` | One class per file (`System.py`, `Thought.py`, … `Symbioid.py`) plus `seed.py`, `formation.py`, `ids.py` |
| `symbioid/core.py` | Thin re-export of `Core` (backward compatible) |
| `main.py` | Smoke demo (six-sets print on completion) |
| `pong_demo.py` / `tetris_demo.py` | Interactive pygame demos |
| `build_demos.py` | PyInstaller packaging for demos |
| `tests/test_core.py` | Unit tests |

## Suggested layout

```text
symbioid/
  .venv/                 # local virtualenv (gitignored)
  README.md
  main.py
  symbioid/              # package: core ontology
  tests/
  notes/                 # freeform markdown / sketches
  experiments/           # throwaway scripts
```

## Related vault / projects

- [[Projects/Antelligence-PoC]] — structured PoC under `antelligence-poc`
- [[Projects/MassAI]] — writing SoT on MT_DATA; design-first
- Antelligence Architecture.docx on MT_DATA under `WRITING/Antelligence Architecture/`
- Neuron/Thought/Signal research: `Work-Log/2026-07-22-research-loop-neuron-thought-signal-trident.md`

## Policy

- **BrainSim / UKS:** cite only; **never** `git commit` / `git push` BrainSim from this tower.
- Prefer **stdlib-first** experiments; add deps only inside `.venv` and pin when something graduates to PoC.

## License

Personal experiment space (MooreTech).
