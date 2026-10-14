# Bioassay Experimental Protocol & Cohort Naming Rules

## 1. Post-2h Starvation Rule (Time Zero Exclusion)
- **Experimental Protocol**: Bees are starved for 2 hours before the treatment application.
- **Rule**: All calculations, mortality kinetics, lethal time estimations (LTx), Log-Rank tests, and Cox Proportional Hazards regression MUST start **after the 2-hour starvation period**.
- **Time Zero Exclusion**: Data from time zero ($t=0$, `d0`, `Day 0`) represents pre-starvation setup baseline and must **NEVER** be included in the statistical analysis.
- The first experimental evaluation day is `d1` (Day 1, after 2h starvation).

## 2. Agreed Standard Cohort & Code Names
Always use the exact 4 agreed cohort names and corresponding code names:
1. **`sucrose solution 2 microliter`** (Code: `Sucrose 2 µL` | Control, Baseline $HR = 1.0$)
2. **`5 % acetone 2 microliter`** (Code: `5% Acetone 2 µL`)
3. **`10 % acetone 2 microliter`** (Code: `10% Acetone 2 µL`)
4. **`5 % acetone 5 microliter`** (Code: `5% Acetone 5 µL`)

*Never use generic labels like "sample", "10 micro liter", or "50uM". Always use the full agreed names and clean code labels.*

## 3. Resurrected Bee Observation Correction Rule
- If a bee is recorded as Dead (`D`) on Day $k$ but recorded as Alive (`A` or `B`) on a subsequent Day $j > k$, the later alive record takes precedence.
- A bee is considered deceased at time $d^*$ only if all states from day $d^*$ through the final observation day are recorded as Dead (`D`).
