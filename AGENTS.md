# Antigravity Bee Analytics Studio Agent Guidelines & Bioassay Protocol

## 🔬 Core Bioassay Protocol Rules (MANDATORY)

### 1. Post-2h Starvation Rule (Exclusion of Time Zero / d0)

- **Experimental Protocol**: Bees undergo a 2-hour starvation period prior to treatment exposure.
- **Critical Rule**: All calculations, mortality rates, lethal time estimations (LT10, LT20, LT50, etc.), Probit linear regression, Kaplan-Meier product-limit curves, Log-Rank tests, and Cox Proportional Hazards regression MUST start **strictly after the 2-hour starvation period**.
- **Time Zero Exclusion**: Data from time zero ($t=0$, `d0`, `Day 0`) is pre-starvation baseline and must **NEVER** be included in calculations or plots.
- The first experimental evaluation day is `d1` / Day 1 (1.0 day post-starvation).

### 2. Standard 4 Experimental Cohorts & Agreed Code Names

Always use the exact 4 agreed cohort names and their standardized code names:

1. **`sucrose solution 2 microliter`** (Code Name: `Sucrose 2 µL` | Control / Baseline Reference $HR = 1.0$)
2. **`5 % acetone 2 microliter`** (Code Name: `5% Acetone 2 µL`)
3. **`10 % acetone 2 microliter`** (Code Name: `10% Acetone 2 µL`)
4. **`5 % acetone 5 microliter`** (Code Name: `5% Acetone 5 µL`)

_Never use outdated or ambiguous labels such as "10 micro liter", "50 uM", or raw file names like "sample\_...". Always normalize and display the agreed full names and code names._

### 3. Integration of Single CSV Dataset (Batch A)

- In Multi-CSV Probit Comparison and Survival Statistics, Batch A (from the Single CSV tab / active observation sheet) MUST always be visible, selectable, and integratable.
- Uploaded files must be mapped directly to clean cohort names and code names rather than retaining raw filenames.

### 4. Resurrected Bee Observation Correction Rule

- If a bee is recorded as Dead (`D`) on Day $k$ but subsequently recorded as Alive (`A` or `B`) on Day $j > k$, the later alive state takes precedence.
- A bee is only considered deceased at time $d^*$ if all subsequent observations through the end of the experiment are recorded as Dead (`D`).
