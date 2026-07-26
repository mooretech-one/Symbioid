# symbioid

Personal **experimentation sandbox** for Antelligence / Symbioid architecture ideas (Thought formation, Faces, Rodin doubling, interface grounding, Outerfaces loops, etc.).

| | |
|--|--|
| **Path** | `~/Desktop/Areas/Personal/Symbioid` |
| **Python** | **3.12** via local **`.venv`** (required on this Ubuntu tower — PEP 668) |
| **Sibling PoC** | `~/Desktop/Areas/Personal/antelligence-poc` (runnable PR-0…PR-3 simulator) |
| **Design** | `~/Desktop/Grok/Work-Log/2026-07-22-design-antelligence-python-poc.md` |
| **Manuscript ingest** | `~/Desktop/Grok/Ingest/pages/2026-07-22-antelligence-architecture-331bf911.md` |

This repo is a **scratch lab**, not a replacement for `antelligence-poc`. Promote stable ideas into the PoC (or a future package) when ready.

## Setup (once)

```bash
cd ~/Desktop/Areas/Personal/Symbioid
python3 -m venv .venv          # already created if you used the setup pass
.venv/bin/pip install -U pip
# optional while experimenting:
# .venv/bin/pip install pytest
```

## Everyday use

```bash
cd ~/Desktop/Areas/Personal/Symbioid
source .venv/bin/activate      # or always prefix with .venv/bin/
python -V                      # should be the venv 3.12.x
```

### Agent memory (demos)

On exit, Tetris/Pong save **agent cognition only** — a **lean** snapshot by default:

| Stored | Not stored |
|--------|------------|
| Mind Observations, Actions (**command keys**), valence, Follows/Integrates maps | Board / ball / scores / coach / cipher |
| Registry poles + **Links that touch them** (weights) | Unregistered formation scaffolding, bulk inactive ghosts |
| | Seeds / laws (rebuilt on `Symbioid()`; **merged** on load) |

| Demo | Default file |
|------|----------------|
| Tetris | `~/.local/share/symbioid/tetris_memory.json` |
| Pong | `~/.local/share/symbioid/pong_memory.json` |

```bash
.venv/bin/python tetris_demo.py              # load if present, save lean on quit
.venv/bin/python tetris_demo.py --no-memory
.venv/bin/python tetris_demo.py --reset-memory
.venv/bin/python tetris_demo.py --memory /tmp/my_mind.json
```

API (`symbioid.persist`): `export_memory(host, mode="lean"|"full")`, `save_memory`, `try_load_into`.

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
| `mind` | `Mind` — recognition / habituation / valence (see below) |
| `innerface` | `Innerface` |
| `interface` | `Interface` |
| `outerface` | `Outerface` |

Pole roles "System"/"Environment" inside the six-seed are **Thoughts**, not the Symbioid instance.

### Mind recognition (growth under constant I/O)

`Mind` is the policy owner that turns “every sample → new Observation” into **same input → same Thought** (or skip):

| Decision | When | Graph effect |
|----------|------|----------------|
| **mint** | Novel content key for that sensor | New content-addressed Observation + sense six-set |
| **reuse** | Known key, not yet habituated | Same Observation poles; no new scaffolding |
| **skip** | Same key ≥ `habituate_after` times in a row | No handoff |

**Follows / FollowedBy** co-occurrence uses the same mint/reuse/skip policy via `Mind.admit_follows`: undirected pair of Observation content keys → stable `sync_id`. Repeated multi-sensor batches with the same values do **not** re-mint Follows scaffolding.

**Integrates / IntegratedBy** (Rodin halving) uses `Mind.admit_integrates`: undirected pole pair **+ awareness channel** (+ depth parent ids for depth-fold) → stable `integrate_id`. Same pair/channel does not remint integrate scaffolding.

### Spiking engines (migration)

`engines_mode` on `Symbioid`: **`legacy`** (default) | `hybrid` | `spike`.

- **Phase 0:** `SpikingEngine` base + `pulse_partition(membership=…)`; `pulse_tick()` = full-graph partition.
- **Phase 1:** `Interface` **is** a `SpikingEngine`.
  - `legacy`: sample → handoff every formation → full `pulse_tick`.
  - `hybrid`/`spike`: sample → stimulate → **membership pulse** → handoff **mint only**.
- **Phase 2:** `Innerface` **is** a `SpikingEngine`.
  - `legacy`: inbox formation/sync/integrate automata as before.
  - `hybrid`/`spike`: **port-in** from Interface → membership pulse → **co-fire consolidator** → sparse depth-fold/prune.
- **Phase 3:** `Outerface` **is** a `SpikingEngine`.
  - `legacy`: Beliefs + graph recommend + law-gated fire.
  - `hybrid`/`spike`: **port-in** from Innerface → membership pulse → **hottest Action** under law gate.
- **Phase 4:** `engines_mode="spike"` demotes inbox automata.
  - Interface: **no** Innerface inbox handoffs; mints go to `structure_pending`.
  - Innerface: pulls `structure_pending` consolidator only (drops inbox); co-fire + sparse prune.
  - Outerface: spike heat first; beliefs last resort.
  - `Symbioid.run_engines()` runs serial **I → N → O**.
- **Phase 5 (now):** port queues + energy budgets + Port-only cross-engine Hebb.
  - `PortPacket` FIFO on `Symbioid.port_queues` (`interface>innerface`, `innerface>outerface`).
  - Engines **export** firers via `export_port_packets`; **import** via `drain_port` + `apply_port_packets`.
  - `Link.is_port=True` channels modulate transfer gain and take Hebb on successful import; they **never** participate in pulse spread.
  - Under membership, non-Port Hebb only when the target is also in membership (`Mind.port_hebb_cross_only`).
  - Per-engine **energy budget** caps fire/spread cost (`Mind.energy_budget_*`, `energy_fire_cost`, `energy_spread_cost`; `0` = unlimited).

```python
s = Symbioid()
s.engines_mode = "spike"
s.mind.energy_budget_interface = 8.0   # optional cap
s.run_engines()   # I → N → O via port queues
```

### Thought firing + decay (Thoughts double as neurons)

There is no separate `Neuron` class. Each **Thought** (including **Link**) carries:

| Field | Role |
|-------|------|
| `activation` | Short-term energy (Signal) |
| `threshold` / `try_fire` | Fire when activation ≥ threshold |
| `decay_rate` / `decay_step` | Leak toward `resting` each pulse tick |
| `refractory_ticks` | Blocks immediate re-fire |

`Symbioid.pulse_tick()`: decay hot set → collect fires → **one-hop** spread along outgoing Links (`weight × propagate_gain`) → **Hebbian** weight update.  
Interface **stimulates** Sensor + Observation poles on sample (mint/reuse/skip).  
Structural seeds / laws use high `threshold` so they do not thrash.

**Activation-based forgetting:** decay alone only quiets energy; structure stays. With **`Mind.forget_cold_enabled=True`** (default), `Innerface.forget_cold_thoughts()` removes **unprotected** Thoughts that have been near resting for ≥ `forget_cold_cycles` host pulse cycles since `last_hot_cycle` (must have been hot at least once). Default **`forget_transient_only=False`** (any unprotected Thought; set `True` to limit to transients). Mind registry Observations/Actions, active six-sets, seeds, laws, last obs, and beliefs stay protected (same set as structure prune).

```python
s.mind.forget_cold_enabled = True   # default
s.mind.forget_cold_cycles = 64
# runs with structure prune via Innerface.maybe_gc()
# s.mind.forget_cold_enabled = False  # opt out
```

**Plasticity (`Link.weight`):**

| Event | Effect |
|-------|--------|
| Both poles fire same tick | `weight += hebb_lr × hebb_co_fire_scale` (clamped) |
| Pre fires, post active | smaller `hebb_pre_post_scale` bump |
| `record_outcome` reward | `reinforce_edge` / `ensure_reciprocal_links` both ways |

Defaults: `weight` starts at 1.0, range **`weight_min`…`weight_max`** (0.05…4). When `weight × propagate_gain ≳ threshold`, one pole can **recruit** its mate (dynamically stronger six-set).

```python
s.mind.dynamics_enabled = True   # master switch
s.mind.hebb_enabled = True
s.stimulate(obs, 1.2)
s.pulse_tick()                   # fire + spread + Hebb + decay clock
```

### Behavior from minted Thoughts

Minted structure is **read for action choice**, not only stored:

| API | Role |
|-----|------|
| `ensure_action_thought(domain, token)` | Stable Action poles (`act:tetris:hard`) |
| `record_outcome(state_poles, token, reward=…)` | Link state ↔ action (Follows + policy Integrates) + valence |
| `recommend_action(state_poles, domain=…)` | Highest-valence Action linked to current state (fail open → `None`) |

**Outerface** prefers `propose_actions_from_graph()` before the legacy “fire first actuator” path.  
**Tetris (v0.0.32+)** is **Symbioid-primary** (`TetrisCoach.network_primary=True`):

1. **Placement / strategy** — `choose_target` **co-leads** cell-map Thought heat and coach board value (`graph_placement_weight≈0.60`; net-primary floor 0.35).
2. **Commands** — `graph_preferred_intent` derives micro-intents from the network-scored target (geo), with Mind `recommend_action` as strong-score override; `tick` takes that intent at high `graph_bias` (~0.93).
3. **Coach retains** — secret-byte discovery, gravity separation, stuck/force-hard survival, cold explore fallback (`last_network_cmd` marks who drove the last byte).
4. **Lock credit** — coach `board_quality_reward` fans valence onto **landing cells** (`apply_lock_valence_to_landing_cells`) so placement heat can improve with experience.
5. **Hole + well avoidance** — `pose_hole_features` / edge-aware `well_metrics`: sealed holes (`d_holes`) and open single-width trenches (`d_well`, `max_well`, including col 0/9). HUD shows `holes` / `well` / `maxW`.

Content keys quantize float `reading` (`quantize_decimals=3` default). Registry Observations + Actions are protected from prune. Coach lock reward also fans into valence via `mind.note_valence(channel="board", delta=…)`.

```python
s = Symbioid()
s.mind.recognition_enabled = True   # default
s.mind.habituate_after = 2
# After sensors sample + outcomes:
# rec = s.mind.recommend_action(poles, domain="tetris")
# Legacy always-mint growth (Observations + Follows):
# s.mind.recognition_enabled = False
```

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

Sensor Thoughts are **stable** (`{sym_id}:sensor:{sensor_id}`); each Input creates a new Observation. Multi-sensor batches also get a lateral six-set (Follows, then optional Integrates).

### Relationship types

Every edge is a `Link`: **Source → [LinkType] → Target** (all Thoughts). Six-sets mint **reciprocal** LinkType pairs. Full catalog: vault `Work-Log/2026-07-25-research-loop-symbioid-relationship-types.md`.

| Forward | Reverse | Role |
|---------|---------|------|
| **Perceives** | **PerceivedBy** | Sensor ⇄ Observation (sense / Input formation) |
| **Follows** | **FollowedBy** | Observation ⇄ Observation (lateral co-occurrence) |
| **Integrates[reason:channel]** | **IntegratedBy[…]** | Observation ⊕ Observation (Rodin halving; v0.0.31+) |
| **Expects** | **ExpectedBy** | Feedback ⇄ expected Observation (Beliefs) |
| **Has** | **IsPartOf** | Host ⇄ Sensor/Actuator (awareness; integrate terminators) |
| **ExistsIn** | **ExistsAround** | System ⇄ Environment (twin seed) |

**Integrates reasons** (live Innerface/Mind): `follows`, `temporal`, `depth`, `cofire`, default `pair`, plus `policy` on outcome registry / tests (e.g. `Integrates[follows:eye]`).

**Also on the graph (not six-set operators):**

| LinkType | Role |
|----------|------|
| **Port** (`Port[channel]` links, `is_port`) | Cross-engine transfer / Hebb (I→N, N→O) |
| **Associates** | Policy plasticity edges (state ↔ Action) |
| **MustPreserveTwinIntegrity**, **MustNotHarm**, **MustObeyUnlessHigher**, **MayPreserveSelfUnlessHigher** | Constitution L0–L3 |

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
cd ~/Desktop/Areas/Personal/Symbioid
source .venv/bin/activate
# optional: .venv/bin/pip install pytest pygame
PYTHONPATH=. python main.py          # sin/cos hand feedback
PYTHONPATH=. python pong_demo.py     # Pong: Symbioid drives both paddles
PYTHONPATH=. python tetris_demo.py   # Tetris: placement learner (feature weights)
PYTHONPATH=. python pong_demo.py --verbose    # six-set / event console dumps
PYTHONPATH=. python tetris_demo.py --verbose
PYTHONPATH=. python -c "from symbioid import Symbioid; print(Symbioid())"
PYTHONPATH=. python -m pytest -q
```

Console dumps are **off by default** (`set_console_emit(False)`). Pass `--verbose` / `-v` for formation and coach logs. HUDs always show a live **Thoughts** count (Tetris also splits active vs inactive and plots them over 1024 turns).

## Demo interface configurations (Sensors / Actuators)

Both demos build a `Symbioid` host, attach `Sensor.transfer` callables that read a physics `world`, sample on a fixed frame cadence, and hand off formations to Interface → Innerface. Game policies (paddle intercept / Tetris coach) write **actuator outputs**; the world is stepped from those values (or mirrored bytes). Each `add_sensor` / `add_actuator` also installs **awareness** six-sets (`Agent Has …`) as integration terminators.

Shared face settings in both demos:

| Setting | Value | Effect |
|---------|--------|--------|
| `interface.continuous_inputs` | `False` | Sample only when the demo calls `sample_into_symbioid` |
| `outerface.wait_for_feedback` | `False` | Do not block the face loop on Feedback |
| Sample cadence | every **4** frames | One formation batch per sample tick |

### Pong (`pong_demo.py`)

Host label: `pong-learner`. Policy: `DualPaddleCoach` (intercept + LMS) writes paddle Y; Symbioid sensors track ball and tracking error.

**Sensors (4)**

| Label | Transfer (from `PongWorld`) | Range / notes |
|-------|----------------------------|---------------|
| `ball_y` | `world.ball_y` | ≈ [−1, 1] playfield Y |
| `ball_vy` | `world.ball_vy * 20` | scaled vertical velocity (readability) |
| `left_err` | `ball_y − left_y` | left paddle tracking error |
| `right_err` | `ball_y − right_y` | right paddle tracking error |

**Actuators (2)**

| Label | Initial `output` | `output_step` | Role |
|-------|------------------|---------------|------|
| `left` | `0.0` | `0.02` | Left paddle Y in [−1, 1] |
| `right` | `0.0` | `0.02` | Right paddle Y in [−1, 1] |

**Loop**

```text
PongWorld ──sensor_world()──► Sensor.transfer ──sample──► Interface → Innerface
    ▲
    │  set_paddles(left.output, right.output)
DualPaddleCoach.control ──writes──► actuators left / right
```

Sample world map includes actuator values (`w["left"]`, `w["right"]`) so transfers and future feedback can close the loop. Learning of intercept (bias, gain, `vy_trust`) lives in `symbioid/world/paddle_learn.py`, not in Outerface action selection.

### Tetris (`tetris_demo.py`)

Host label: `tetris-byte-learner`. Policy: `TetrisCoach` discovers a secret byte→command map, then places pieces (1-ply sim + learned board value). Control is a **single normalized byte**; the world cipher maps only a few of 0–255 to left/right/rotate/hard.

**Sensors (204)** — full spatial map + slim meta

| Label | Transfer (from `TetrisWorld`) | Scaling |
|-------|------------------------------|---------|
| `cell_r{RR}_c{CC}` × **200** (10×20) | `cell_reading(r,c)` with active piece painted | **1.0** block · **0.5** hole · **0.0** open |
| `piece_id` | active piece kind | index in I…L `/ 6` → [0, 1] |
| `next_id` | next piece kind | index `/ 6` → [0, 1] |
| `lines` | lines cleared this game | `min(1, lines/50)` |
| `last_byte` | last control byte applied | `last_byte / 255` → [0, 1] |

Hole = empty cell with a filled cell above in the same column (classic Tetris). Aggregate height/hole totals are no longer separate sensors; the map supersedes them.

**Perf:** cell sensors use `awareness=False` (terminator only). Sampling is **change-only** + **ROI** (v0.0.28): empty top rows (`sky_row`), solid full-width floor (`solid_floor_start_row`), active dirty-rect, sticky locked=1.0, line-clear invalidation; candidates via `(r,c)` index. **Dynamics** (v0.0.29): Link adjacency index + hot-set-only `pulse_tick`; demo pulses every **2** frames; Phase C cell-obs index. **Learning structure** (v0.0.30): cross-channel Integrate block deactivates Follows sync; `max_active_syncs`/`max_active_senses` LRU; Tetris `cofire_meta_only` + no cell–cell Follows; hard Mind follows/integrates registry caps. **Band B** (Tetris demo): `max_active_senses=224`, `max_active_syncs=112`, `max_active_integrates=112`; Mind registries **4096** with **policy-first** hard eviction (keep `act:` associations); HUD shows `se/sy/in` + `fl/ig` registry.

**Actuators (1)**

| Label | Initial `output` | `output_step` | Role |
|-------|------------------|---------------|------|
| `byte` | `0.0` | `1/255` | Command register ∈ [0, 1]; maps to integer byte via `× 255` |

**Loop**

```text
TetrisWorld ──sensor_world()──► Sensor.transfer ──sample──► Interface → Innerface
     ▲
     │  step_byte(code)          cipher: few live bytes → left|right|rotate|hard
TetrisCoach.tick ──writes──► actuator "byte".output = code/255
```

The coach does **not** use `Actuator.request_fire` for moves; it writes the byte channel and steps the world directly. Symbioid still forms Thoughts/Beliefs from board sensors. Placement search uses world physics (`simulate_placement`) scored by a height-shaped + learned evaluator; the secret control map is learned from observation only (never from reading `cipher`).

**UI extras (Tetris):** pause after top-out for Innerface catch-up; plots of **Active**, **Inactive**, and **Minted** Thought counts over a 1024-turn window; highscores sorted best-first (`#ddd ssssss`). Superseded Rodin sense/sync scaffolding (Links + relation types) is GC'd from the live graph after integrate/depth-fold.

## Single-file executables (Pong + Tetris)

Package each demo with **PyInstaller** into one binary per OS/arch (no Python install needed on the target machine).

| Output | Source |
|--------|--------|
| `dist/SymbioidPong` (`.exe` on Windows) | `pong_demo.py` |
| `dist/SymbioidTetris` | `tetris_demo.py` |
| `dist/SymbioidPong-linux-x64` etc. | tagged copies for multi-OS drops |

### Build on this machine

```bash
cd ~/Desktop/Areas/Personal/Symbioid
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
