# Solver-Agnostic Simulation: Detailed Refactoring Plan

## Objective

Extract device initialization and simulation from [`GeneticSimulation`](src/akkudoktoreos/optimization/genetic/genetic.py:92) into a **solver-independent** class that can be reused by any optimizer (GA, DP, MPC, LP).

---

## 1. Current Situation

### Current Architecture (main branch)

```
GeneticOptimization (genetic.py:571)
├── GeneticSimulation (genetic.py:92)        ← Device simulation, GA-bound
│   ├── _simulate_init()                     ← Device-Setup + Array-Config
│   ├── _simulate_hourly_step()              ← Per-hour computation
│   ├── _simulate_finalize()                 ← numpy-conversion + aggregation
│   └── simulate()                           ← Orchestrator
├── evaluate()                               ← Fitness + Penalties
├── optimize()                               ← DEAP GA-Loop
└── optimize_ems()                           ← Entry point, Device-Init inline
```

**Problem:** `GeneticSimulation` is deeply embedded in `genetic.py` and has GA-specific dependencies (DEAP, GA-encoding). New solvers (MPC, LP) must either:
- Duplicate the same simulation (as DP initially did)
- Or use `GeneticSimulation` as a black box (poorly testable/extensible)

### Branch `feature/alternative-mpc-solver` – Existing Hints

The branch already contains extensive analysis and partial implementation:

| File | Purpose | Status |
|------|---------|--------|
| [`plans/dp-ga-code-reuse-plan.md`](plans/dp-ga-code-reuse-plan.md) | 6-phase refactoring plan | Draft |
| [`plans/dp-prototype.py`](plans/dp-prototype.py) | DP PoC (~200 lines) | Working |
| `optimization.py` | Shared settings + `OptimizationSolution` | Implemented |
| `genetic_setup.py` | Device-Init + Charge-Rates + Action-Decoding + Pure Physics | Implemented |
| `genetic_penalties.py` | AC break-even, EV SOC miss, Battery residual value | Implemented |
| `dp/dp.py` | Full DP implementation | Implemented |
| `dp/dpsolution.py` | Delegates to `GeneticSolution` | Implemented |

**Assessment:** The branch already extracted shared modules (`genetic_setup.py`, `genetic_penalties.py`). Our plan builds on top of that and goes one step further: making the **simulation itself** solver-agnostic.

---

## 2. Guiding Principles

1. **Simulation ≠ Optimization**: The simulation computes energy flows for a given action sequence. It knows NOTHING about fitness, mutation, backward-pass, or simplex.
2. **Device-Init ≠ Device Physics**: Device objects (Battery, Inverter, EV) are created once. Pure physics (SOC transition) is stateless.
3. **Pure Functions for DP**: DP needs stateless, vectorizable functions for the backward pass (~100k+ state-action combinations).
4. **Mutable State for GA**: GA needs fast, mutable state updates per individual.
5. **Constraints for LP/MPC**: LP/MPC solvers need linear/smooth constraint functions – no if/else logic.

---

## 3. Target Architecture

```
akkudoktoreos/optimization/
├── optimizationabc.py              # OptimizationBase (existing)
├── optimization.py                 # Common settings + OptimizationSolution (from branch)
│
├── simulation/                     # ★ NEW: Solver-agnostic simulation
│   ├── __init__.py
│   ├── devices.py                  # DeviceFactory + SimulationDevices NamedTuple
│   ├── physics.py                  # Pure functions: compute_next_soc(), etc.
│   ├── step.py                     # EnergySimulationStep dataclass
│   ├── engine.py                   # EnergySimulationEngine (main class)
│   └── result.py                   # SimulationResult (existing GeneticSimulationResult)
│
├── penalties/                      # ★ NEW: Shared penalty module
│   ├── __init__.py
│   └── penalties.py                # AC break-even, EV SOC miss, residual value
│
├── genetic/
│   ├── genetic.py                  # GeneticOptimization + DEAP (slimmed down)
│   ├── geneticparams.py            # (unchanged)
│   ├── geneticsolution.py          # (unchanged)
│   ├── geneticdevices.py           # (unchanged)
│   └── encoding.py                 # encode/decode, split/merge (extracted from genetic.py)
│
└── dp/
    ├── __init__.py
    ├── dp.py                       # DPOptimization (uses EnergySimulationEngine)
    └── dpsolution.py               # (from branch)
```

---

## 4. Detailed Class Designs

### 4.1 `EnergySimulationEngine` (Main Class)

**Location:** `src/akkudoktoreos/optimization/simulation/engine.py`

**Responsibilities:**
1. Device initialization from parameters + config
2. Action sequence → simulation → result
3. Per-hour computation (load, EV, battery SOC, inverter, financials)
4. Result aggregation (numpy arrays, totals)

**Design:**

```python
@dataclass
class SimulationConfig:
    """Immutable configuration for a simulation run."""
    prediction_hours: int
    optimization_hours: int
    start_hour: int = 0
    # Forecast data
    load_energy_array: np.ndarray = field(default_factory=np.zeros)
    pv_prediction_wh: np.ndarray = field(default_factory=np.zeros)
    elect_price_hourly: np.ndarray = field(default_factory=np.zeros)
    elect_revenue_per_hour: np.ndarray = field(default_factory=np.zeros)
    temperature_forecast: np.ndarray = field(default_factory=np.zeros)


class EnergySimulationEngine:
    """Solver-agnostic energy simulation engine.

    Takes device parameters + forecast data + action sequence,
    returns simulation results (costs, grid energy, SOC trajectory, etc.).

    Usage:
        engine = EnergySimulationEngine.create(parameters, config)
        result = engine.run(ac_charge, dc_charge, discharge, ev_charge, appliance_start)
    """

    # ── Factory ──────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        parameters: GeneticOptimizationParameters,
        config: ConfigEOS,
        start_hour: int = 0,
    ) -> "EnergySimulationEngine":
        """Create engine from optimization parameters + global config."""
        # 1. Create devices
        battery = Battery(...)
        ev = Battery(...) if parameters.ev else None
        inverter = Inverter(...)
        home_appliance = HomeAppliance(...) if parameters.home_appliance else None

        # 2. Extract forecast data
        sim_config = SimulationConfig(...)

        return cls(
            battery=battery,
            ev=ev,
            inverter=inverter,
            home_appliance=home_appliance,
            sim_config=sim_config,
        )

    # ── Public API ───────────────────────────────────────────────────
    def run(
        self,
        ac_charge: np.ndarray,
        dc_charge: np.ndarray,
        discharge: np.ndarray,
        ev_charge: Optional[np.ndarray] = None,
        home_appliance_start: Optional[int] = None,
    ) -> SimulationResult:
        """Run full simulation for given action sequence."""
        # Phase 1: Init – reset devices, configure arrays
        self._init(ac_charge, dc_charge, discharge, ev_charge, home_appliance_start)
        # Phase 2: Hourly steps
        self._step_all()
        # Phase 3: Finalize – convert to result
        return self._finalize()

    def step(
        self,
        hour: int,
        ac_charge: float,
        dc_charge: float,
        discharge: int,
        ev_charge: float = 0.0,
    ) -> EnergySimulationStep:
        """Run single hourly step. For iterative solvers (MPC)."""
        ...

    # ── Constraints (for LP/MPC) ────────────────────────────────────
    def get_battery_soc_constraints(
        self,
    ) -> list[tuple[str, np.ndarray, float]]:
        """Return linear constraints for battery SOC.

        Returns list of (name, coefficients, rhs) tuples suitable for
        scipy.optimize.linprog or cvxpy.
        """
        ...

    def get_inverter_power_constraints(self) -> list[tuple[str, np.ndarray, float]]:
        """Return linear constraints for inverter power limits."""
        ...

    def get_grid_balance_constraints(self) -> list[tuple[str, np.ndarray, float]]:
        """Return power balance constraints (generation = consumption + storage)."""
        ...

    # ── Internal ─────────────────────────────────────────────────────
    def _init(...) -> None: ...
    def _step_all(...) -> None: ...
    def _finalize(...) -> SimulationResult: ...
```

### 4.2 `EnergySimulationStep` (Per-Hour Result)

**Location:** `src/akkudoktoreos/optimization/simulation/step.py`

```python
@dataclass(frozen=True)
class EnergySimulationStep:
    """Immutable result of a single hourly simulation step."""
    # Energy flows [Wh]
    consumption: float           # Total consumption from grid + local
    energy_feedin_grid: float    # Energy fed to grid
    energy_consumption_grid: float  # Energy drawn from grid
    losses: float               # Total losses (EV + inverter + battery)
    self_consumption: float     # PV self-consumption
    home_appliance_wh: float    # Home appliance energy this hour

    # Financial [Euro]
    cost: float                 # Grid import cost
    revenue: float              # Grid export revenue
    electricity_price: float    # Current electricity price
```

### 4.3 `physics.py` – Pure Functions

**Location:** `src/akkudoktoreos/optimization/simulation/physics.py`

**Purpose:** Stateless, vectorizable physics functions for DP backward-pass and LP constraints.

```python
def compute_battery_next_soc(
    current_soc_wh: float,
    ac_charge_factor: float,
    dc_charge_allowed: bool,
    discharge_allowed: bool,
    pv_wh: float,
    load_wh: float,
    *,
    # Battery params
    min_soc_wh: float,
    max_soc_wh: float,
    charging_efficiency: float,
    discharging_efficiency: float,
    max_ac_charge_power_w: float,
    ac_to_dc_efficiency: float = 1.0,
    dc_to_ac_efficiency: float = 1.0,
) -> float:
    """Pure function: next battery SoC after one time step."""
    ...

def compute_ev_next_soc(
    current_soc_wh: float,
    charge_factor: float,
    *,
    min_soc_wh: float,
    max_soc_wh: float,
    charging_efficiency: float,
    max_charge_power_w: float,
) -> float:
    """Pure function: next EV SoC after one time step."""
    ...

def compute_energy_balance(
    load_wh: float,
    pv_wh: float,
    bat_charge_wh: float,
    bat_discharge_wh: float,
    ev_charge_wh: float,
    home_appliance_wh: float,
) -> tuple[float, float]:
    """Compute grid import/export from energy balance.

    Returns (grid_import_wh, grid_export_wh).
    """
    ...

def compute_hourly_cost(
    grid_import_wh: float,
    grid_export_wh: float,
    price_per_wh: float,
    revenue_per_wh: float,
) -> tuple[float, float]:
    """Compute cost and revenue for one hour.

    Returns (cost, revenue).
    """
    ...
```

**Important:** These functions already exist in the branch (`genetic_setup.py`). They will be moved to `simulation/physics.py`.

### 4.4 `devices.py` – DeviceFactory

**Location:** `src/akkudoktoreos/optimization/simulation/devices.py`

```python
@dataclass(frozen=True)
class SimulationDevices:
    """Immutable container for all simulation devices."""
    battery: Battery
    ev: Optional[Battery]
    inverter: Inverter
    home_appliance: Optional[HomeAppliance]


class DeviceFactory:
    """Factory for creating simulation devices from parameters + config."""

    @staticmethod
    def create_devices(
        parameters: GeneticOptimizationParameters,
        config: ConfigEOS,
        prediction_hours: int,
    ) -> SimulationDevices:
        ...
```

---

## 5. Migration Plan

### Phase 1: Create `simulation/` Package

**Goal:** New `simulation/` structure with empty shells.

**Steps:**
1. Create `src/akkudoktoreos/optimization/simulation/__init__.py`
2. Create `physics.py` – copy pure functions from `genetic_setup.py` (branch)
3. Create `step.py` – `EnergySimulationStep` dataclass
4. Create `result.py` – copy `GeneticSimulationResult` from `geneticsolution.py`
5. Create `devices.py` – `DeviceFactory` + `SimulationDevices`
6. Create `engine.py` – `EnergySimulationEngine` skeleton

**Tests:** Unit tests for `physics.py` functions (pure math, no state).

**Risk:** LOW – New code, no changes to existing code.

### Phase 2: Implement `EnergySimulationEngine`

**Goal:** Fully functional engine that replaces `GeneticSimulation.simulate()`.

**Steps:**
1. Implement `EnergySimulationEngine.create()` – device factory
2. Implement `EnergySimulationEngine.run()` – 3-phase simulation
3. Implement `EnergySimulationEngine.step()` – single-step for MPC
4. Verify: Bitwise identical with `GeneticSimulation.simulate()`

**Tests:**
- Identity test: Engine vs. GeneticSimulation on all test data
- Performance test: ≤ 5% overhead vs. GeneticSimulation

**Risk:** MEDIUM – New implementation must produce exactly the same results.

### Phase 3: Migrate `GeneticSimulation` → `EnergySimulationEngine`

**Goal:** `GeneticSimulation` becomes a thin wrapper around `EnergySimulationEngine`.

**Steps:**
1. Replace `GeneticSimulation._simulate_init()` → delegates to engine
2. Replace `GeneticSimulation._simulate_hourly_step()` → delegates to engine
3. Replace `GeneticSimulation._simulate_finalize()` → delegates to engine
4. `GeneticSimulation.simulate()` → `self.engine.run(...)`
5. Remove `_HourlyStepResult`, `_SimulationContext` from `genetic.py`

**Tests:** All existing tests must continue to pass (41 passed).

**Risk:** LOW – Wrapper pattern, existing API preserved.

### Phase 4: Extract GA-Specific Code

**Goal:** `genetic.py` only contains GA logic (DEAP, encode/decode, evaluate).

**Steps:**
1. Create `genetic/encoding.py`:
   - `encode_charge_discharge()` (inverse of `decode_charge_discharge()`)
   - `decode_charge_discharge()` (import from `genetic_setup.py`)
   - `split_individual()`, `merge_individual()`
2. Keep `evaluate()` in `GeneticOptimization` → stays in `genetic.py`
3. Move device-init from `optimize_ems()` → `DeviceFactory.create_devices()`
4. Move charge-rates from `optimize_ems()` → `get_battery_charge_rates()` (exists)

**Tests:** GA tests, encoding/decoding tests.

**Risk:** LOW – Code organization, no logic changes.

### Phase 5: Consolidate Penalty Modules

**Goal:** All penalty functions in `penalties/` package.

**Steps:**
1. Create `src/akkudoktoreos/optimization/penalties/__init__.py`
2. Copy `genetic_penalties.py` (branch) → `penalties/penalties.py`
3. Replace imports in `genetic.py` and `dp/dp.py`
4. Remove `genetic_penalties.py` from `genetic/` package

**Tests:** Penalty unit tests, integration tests.

**Risk:** LOW – Import changes only.

### Phase 6: DP Integration

**Goal:** `DPOptimization` uses `EnergySimulationEngine`.

**Steps:**
1. Merge `feature/alternative-mpc-solver` → main (or cherry-pick)
2. Replace `DPOptimization.simulation` (GeneticSimulation) → `EnergySimulationEngine`
3. Forward-pass: Use `engine.run()` instead of `_hourly_step()`
4. Backward-pass: Use `physics.py` pure functions (unchanged)

**Tests:** DP tests (`test_dp.py`), GA-vs-DP comparison (`benchmark_ga_dp.py`).

**Risk:** MEDIUM – DP integration must be verified.

### Phase 7: Prepare LP/MPC Foundation (optional)

**Goal:** `EnergySimulationEngine` provides constraint methods for LP/MPC.

**Steps:**
1. Implement `get_battery_soc_constraints()` → linear inequalities
2. Implement `get_inverter_power_constraints()` → linear inequalities
3. Implement `get_grid_balance_constraints()` → linear equations
4. Create example: `lp_optimizer.py` with `scipy.optimize.linprog`

**Tests:** LP results vs. GA/DP on small problem instances.

**Risk:** HIGH – New solver type, complex mathematics.

---

## 6. Expected Savings

| Area | Before (genetic.py) | After | Net |
|------|-------------------|-------|-----|
| Simulation code | ~450 lines (genetic.py) | ~300 lines (engine.py) | -150 |
| Device init | ~80 lines (inline) | ~40 lines (devices.py) | -40 |
| Pure physics | 0 (in genetic.py) | ~150 lines (physics.py) | +150 |
| GA code | ~900 lines (total) | ~500 lines (encoding.py + genetic.py) | -400 |
| Penalties | ~150 lines (genetic.py) | ~150 lines (penalties.py) | 0 |
| **Total** | **~1580 lines** | **~1140 lines** | **-440 (-28%)** |

**More importantly:** Code reuse (not just savings):
- DP: ~600 → ~350 lines (uses engine + physics)
- GA: ~1400 → ~700 lines (uses engine + encoding)
- MPC/LP: 0 → ~200 lines (uses engine + constraints)

---

## 7. Backward Compatibility

### API Stability

| API | Status | Remark |
|-----|--------|--------|
| `GeneticSimulation` | **PRESERVED** | Wrapper around engine |
| `GeneticSimulation.simulate()` | **PRESERVED** | Delegates to engine |
| `GeneticOptimization.optimize_ems()` | **PRESERVED** | Uses engine internally |
| `GeneticSolution` | **PRESERVED** | No changes |
| `GeneticSimulationResult` | **IMPROVED** | Moved to `simulation/result.py` |

### Breaking Changes

- `GeneticSimulation` fields (`battery`, `ev`, `inverter`) → accessible via `engine.devices`
- `_HourlyStepResult`, `_SimulationContext` → internal implementation, not part of public API
- `genetic_setup.py`, `genetic_penalties.py` → moved to `simulation/` + `penalties/`

---

## 8. Mermaid Diagram: Target Architecture

```mermaid
graph TB
    subgraph Shared[Shared Simulation Layer]
        Engine[EnergySimulationEngine<br/>run() step() constraints()]
        Physics[physics.py<br/>compute_next_soc() balance() cost()]
        Devices[devices.py<br/>DeviceFactory SimulationDevices]
    end

    subgraph Penalties[Penalty Layer]
        PEN[penalties.py<br/>ac_break_even ev_soc_miss residual_value]
    end

    subgraph Genetic[Genetic Algorithm]
        GO[GeneticOptimization<br/>DEAP evaluate optimize_ems]
        ENC[encoding.py<br/>encode decode split merge]
        GSOL[GeneticSolution<br/>optimization_solution energy_management_plan]
    end

    subgraph DP[Dynamic Programming]
        DO[DPOptimization<br/>backward_pass forward_pass]
        DSOL[DPSolution<br/>delegates to GeneticSolution]
    end

    subgraph Future[Future Solvers]
        MPC[MPCOptimizer<br/>model predictive control]
        LP[LPOptimizer<br/>linear programming]
    end

    subgraph Devices[Device Classes]
        Bat[Battery]
        Inv[Inverter]
        EV[Battery as EV]
        HA[HomeAppliance]
    end

    Engine --> Physics
    Engine --> Devices
    Devices --> Bat
    Devices --> Inv
    Devices --> EV
    Devices --> HA

    GO --> Engine
    GO --> ENC
    GO --> PEN
    GO --> GSOL

    DO --> Engine
    DO --> Physics
    DO --> PEN
    DO --> DSOL
    DSOL --> GSOL

    MPC --> Engine
    MPC --> PEN
    LP --> Engine
    LP --> PEN
```

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Performance regression | MEDIUM | HIGH | Benchmarks after each phase |
| Result deviation | LOW | CRITICAL | Bitwise identity tests |
| Test coverage | LOW | MEDIUM | New tests for simulation/ |
| Breaking changes | MEDIUM | MEDIUM | Wrapper pattern, API preserved |
| DP integration | MEDIUM | MEDIUM | Cherry-pick instead of full merge |

---

## 10. Next Steps (Priority)

1. **[Phase 1]** Create `simulation/` package with empty shells
2. **[Phase 2]** Implement `EnergySimulationEngine` + identity tests
3. **[Phase 3]** `GeneticSimulation` → wrapper around engine
4. **[Phase 4]** Extract GA code (`encoding.py`)
5. **[Phase 5]** Consolidate penalty modules
6. **[Optional]** Phase 6-7: DP integration + LP/MPC foundation
