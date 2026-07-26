# Dynamic Programming (DP) Solver

Bellman-Optimality-basierter Solver für Energieoptimierung mit diskretem State-Space und Numba-JIT-Kompilierung.

## Überblick

Der DP-Solver findet die exakte optimale Lösung im diskretisierten State-Space mittels Backward-Pass mit Bellman-Gleichung, Policy-Storage und Backtracking.

### Features

- **Konfigurierbare SoC-Resolution**: Default 1% (101 Level), 2% verfügbar (51 Level, ~1.3x schneller)
- **Numba-JIT-Kompilierung**: `@njit(cache=True)` für C-Geschwindigkeit (~0.04s statt ~51s)
- **Bellman-Optimalität**: Exakte Lösung im diskretisierten Raum via Backward-Pass
- **EV-Optimierung**: EV-Ladeplanung als State/Action-Dimension
- **Home-Appliance-Scheduling**: Startzeit als DP-Entscheidung
- **Terminal-Penalties**:
  - Battery residual value: Netto-SOC-Änderung × (Grid-Preis + Feed-in)/2 (letzte Stunde)
  - EV residual value: Analog für EV-Ladung mit Strompreis-Bewertung
  - EV SOC miss penalty: Bestrafung wenn EV unter min_soc_percentage
  - AC charge break-even penalty: Ökonomisch nicht gerechtfertigtes AC-Charging
- **GA-Parität**: DC-Charge-Flag, Worst-Case-Mode, EV-Optimierung-Check
- **HYBRID-Mode**: DP als GA-Warmup via `parameters.start_solution` mit mutierten Varianten

## State-Space

```
State = (hour, battery_soc_index, ev_soc_index, appliance_started)
```

- `hour`: 0 bis prediction_hours
- `battery_soc_index`: 0 bis N-1 (N = 100/SOC_RESOLUTION_PERCENT + 1)
- `ev_soc_index`: 0 bis N-1 (gleiche Resolution wie Batterie)
- `appliance_started`: 0 oder 1

Maximale States (2% Resolution, 48h-Horizont): `(48+1) × 51 × 51 × 2 = 253,842`

Maximale States (1% Resolution, 48h-Horizont): `(48+1) × 101 × 101 × 2 = 979,296`

## Action-Space

```
Action = (ac_charge_rate_idx, dc_charge_allowed, discharge_allowed, ev_charge_rate_idx)
```

- `ac_charge_rate_idx`: Index in charge_rates-Array (z.B. [0.0, 0.5, 1.0])
- `dc_charge_allowed`: True/False (PV→Battery direkt)
- `discharge_allowed`: True/False (Battery→Grid/Haus)
- `ev_charge_rate_idx`: Index in EV charge_rates-Array (z.B. [0.0, 0.5, 1.0])

## Bellman-Gleichung

Backward-Pass von t = horizon-1 nach t = 0:

```
V[t,s] = min_a { c(t,s,a) + V[t+1, T(s,a)] }
```

- `V[t,s]`: Optimaler kumulierter Cost-to-go von State s bei Zeit t
- `c(t,s,a)`: Sofortkosten für Action a in State s bei Zeit t
- `T(s,a)`: Next-State nach Action a
- Terminal: `V[horizon, s] = terminal_penalty(s)` (Battery/EV residual value + SOC miss)

## Kostenberechnung

Die Sofortkosten `c(t,s,a)` berücksichtigen:

- Grid-Verbrauch: `grid_import_wh × electricity_price[t]`
- PV-Einspeisung: `-grid_feed_in_wh × feed_in_tariff[t]`
- Verluste: Durch Inverter und Batteriewirkungsgrad korrekt modelliert
- Physik: [`compute_battery_next_soc_with_flows()`](src/akkudoktoreos/optimization/simulation/physics.py:167)

## Terminal-Penalties

### Battery Residual Value

[`battery_residual_value_penalty()`](src/akkudoktoreos/optimization/simulation/penalties.py:13) bewertet Netto-SOC-Änderung:

```python
# Letzten Stundenpreise verwenden (Horizont-Ende)
last_grid_price = float(electricity_prices[-1])
last_feed_in = float(feed_in_tariffs[-1])
avg_price = (last_grid_price + last_feed_in) / 2.0

net_change_wh = battery_energy_content_wh - initial_soc_wh
penalty = -(net_change_wh * dc_to_ac_efficiency * avg_price)
```

- `end_soc > initial_soc`: Penalty für investierte Energie (negativ)
- `end_soc < initial_soc`: Gutschrift für genutzte Energie (positiv)
- Preis: Mittelwert aus Grid-Preis und PV-Einspeisevergütung der letzten Stunde

### EV Residual Value

[`ev_residual_value_penalty()`](src/akkudoktoreos/optimization/simulation/penalties.py:74) analog zur Batterie:

```python
net_change_wh = ev_energy_content_wh - initial_soc_wh
penalty = -(net_change_wh * avg_price)
```

### EV SOC Miss Penalty

[`ev_soc_miss_penalty()`](src/akkudoktoreos/optimization/simulation/penalties.py:132) bestraft wenn EV unter min_soc_percentage:

```python
if ev_soc_percentage < min_soc_percentage:
    return abs(min_soc_percentage - ev_soc_percentage) * penalty_factor
return 0.0
```

## Konfiguration

### Algorithm-Auswahl

In [`config.yaml`](config.yaml) oder [`OptimizationCommonSettings`](src/akkudoktoreos/optimization/optimization.py:58):

```yaml
optimization:
  algorithm: "DP"      # Nur DP
  # algorithm: "GENETIC"  # Nur GA
  # algorithm: "HYBRID"   # DP als GA-Warmup
```

### DP-spezifische Parameter

```yaml
optimization:
  algorithm: "DP"
  horizon_hours: 48
  optimize_ev: true        # EV-Ladeplanung aktivieren
  worst_case: false        # True = Maximiere Kosten (Stress-Test)
  optimize_dc_charge: true # DC-Charge als Entscheidung
```

### SOC-Resolution

In [`DPOptimizer`](src/akkudoktoreos/optimization/dp/dpoptimizer.py:346):

```python
SOC_RESOLUTION_PERCENT = 2.0  # Default: 2% (51 Level)
# SOC_RESOLUTION_PERCENT = 1.0  # Höhere Präzision: 1% (101 Level, ~1.3x langsamer)
```

## API

### DPOptimizer

```python
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.dp.dpparams import DPOptimizationParameters

# Parameter vorbereiten
params = await DPOptimizationParameters.prepare()

# Optimieren
optimizer = DPOptimizer()
solution = optimizer.optimize(
    params=params,
    ha_params=params.dishwasher,
    start_hour=10,
    worst_case=False,
    optimize_ev=True,
    optimize_dc_charge=True,
)

print(f"States explored: {solution.total_states_explored}")
print(f"Time: {solution.computation_time_ms}ms")
print(f"AC charge: {solution.ac_charge}")
```

### HYBRID-Mode (DP → GA)

```python
from akkudoktoreos.optimization.dp.dpoptimizer import DPOptimizer
from akkudoktoreos.optimization.genetic.genetic import GeneticOptimization
from akkudoktoreos.optimization.genetic.geneticparams import GeneticOptimizationParameters

# DP-Lösung finden
dp_solution = dp_optimizer.optimize(params=dp_params, ...)

# Als GA-Individual konvertieren
ga_individual = dp_optimizer.to_ga_individual(dp_solution)

# GA mit warmup starten (via start_solution Parameter)
ga_params = await GeneticOptimizationParameters.prepare()
ga_params.start_solution = ga_individual

ga_solution = ga_optimizer.optimize_ems(
    parameters=ga_params,
    start_hour=10,
)
```

Im HYBRID-Mode wird die DP-Lösung als Elite-Individual verwendet + 9 mutierte Varianten (±10% AC-Charge, ±2 EV-Charge, ±5h Washingstart) für die initiale GA-Population.

## Vergleich: DP vs GA vs HYBRID

| Feature | GA | DP | HYBRID |
|---------|-----|-----|--------|
| Lösungstyp | Approximiert (evolutionär) | Exakt (diskretisiert) | DP als Warmup + GA-Verfeinerung |
| Laufzeit | 1.8-2.0s | 0.17-0.66s | 2.1-2.2s |
| Reproduzierbar | Mit Seed | Ja (deterministisch) | Mit Seed |
| State-Space | Nicht relevant | Vollständig erkundet | DP erkundet, GA verfeinert |
| Terminal-Penalties | Ja (solver-agnostic) | Ja (solver-agnostic) | Ja (beide) |
| Worst-Case | Ja | Ja | Ja |

## Benchmark

### optimize_input_2.json (48h-Horizont, mit EV)

| Solver | Balance | Time | Battery End SOC |
|--------|---------|------|-----------------|
| GA | 8.1519€ | 1.78s | 33.93% |
| DP | **8.8352€** | 0.66s | 42.98% |
| HYBRID | 7.8057€ | 2.13s | 14.18% |

DP erreicht das beste Ergebnis mit korrekter Bewertung von Batterie- und EV-Energie.

### Skripte

```bash
# Alle Solver, alle Testdaten
uv run tmp/scripts/benchmark_all_data.py

# Vergleich GA/DP/HYBRID
uv run tmp/scripts/benchmark_solvers.py
```

## Tests

```bash
# Alle DP-Tests
uv run pytest tests/test_dpoptimize.py -v

# Einzelne Tests
uv run pytest tests/test_dpoptimize.py::test_dp_optimize_basic -v
uv run pytest tests/test_dpoptimize.py::test_dp_vs_ga_comparison -v
uv run pytest tests/test_dpoptimize.py::test_dp_ev_optimization -v
uv run pytest tests/test_dpoptimize.py::test_dp_hybrid_mode -v
uv run pytest tests/test_dpoptimize.py::test_dp_worst_case_mode -v
```

## Performance-Optimierung

### Numba-JIT Backward-Pass

[`_bellman_backward()`](src/akkudoktoreos/optimization/dp/dpoptimizer.py:210) ist mit `@njit(cache=True)` dekoriert für C-Geschwindigkeit:

- Original: ~51s (reines Python)
- Mit Precompute: ~20s
- Mit Numba-JIT: ~0.04s (1250x schneller)

### Vectorisierte Terminal-Penalties

Terminal-Penalties werden vektorisiert über alle SOC-Level berechnet:

```python
V[horizon, :, :, ap_started] = (
    bat_resid_values[:, np.newaxis]
    + ev_penalties[np.newaxis, :]
    + ac_penalty_base
)
```

## Referenzen

- Bellman, R. (1957): Dynamic Programming
- [`plans/dynamic-programming-solver-plan.md`](plans/dynamic-programming-solver-plan.md)
- [`src/akkudoktoreos/optimization/simulation/penalties.py`](src/akkudoktoreos/optimization/simulation/penalties.py) - Solver-agnostic Penalty-Funktionen
- [`src/akkudoktoreos/optimization/simulation/physics.py`](src/akkudoktoreos/optimization/simulation/physics.py) - Batteriewirkungsgrad und Energieflüsse
