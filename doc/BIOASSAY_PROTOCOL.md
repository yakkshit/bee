# 🔬 Honeybee Bioassay Experimental Protocol & Ground Rules

This document specifies the standard bioassay experimental design, data integrity rules, and observational correction principles enforced by **Bee Analytics Studio**.

---

## 1. Post-2h Starvation Rule (Exclusion of Time Zero / $d_0$)

### Experimental Procedure
Before honeybees (*Apis mellifera*) receive treatment exposure, they undergo a mandatory **2-hour pre-experimental starvation period**:
1. Bees are collected from healthy hives in the morning.
2. They are gently sorted into experimental cages (10 bees per cage).
3. All feeders are withheld for 2.0 hours at $30^\circ\text{C}$ and $65\%$ relative humidity.
4. This ensures bees deplete stomach reserves, synchronize hunger levels, and feed immediately once treatment tubes are installed.

### Mathematical & Computational Rule
- Any data recorded at time zero ($t = 0$, `d0`, `Day 0`) reflects cage-handling stress, transportation mortality, or baseline equipment tare.
- **Strict Exclusion**: All probit models, lethal time estimations ($LT_{10} \dots LT_{90}$), Kaplan-Meier survival curves, Log-Rank tests, and Cox regression MUST start **strictly after the 2-hour starvation period**.
- Pre-starvation baseline data ($t=0$) is **never** included in survival or mortality models.
- The first experimental evaluation day is **Day 1** ($d_1 = 1.0\text{ day}$ post-starvation).

---

## 2. Standard 4 Experimental Cohorts & Nomenclature

In toxicology trials, raw filenames and shorthand codes can cause confusion across laboratories and blinded reviewers. Bee Analytics Studio normalizes all data into 4 standardized experimental cohorts:

| Standard Cohort Name | Publication Code Name | Role / Concentration | Reference Status |
| :--- | :--- | :--- | :--- |
| **`sucrose solution 2 microliter`** | `Sucrose 2 µL` | Pure 50% w/w Sucrose control | **Baseline Reference ($HR \equiv 1.0$)** |
| **`5 % acetone 2 microliter`** | `5% Acetone 2 µL` | Solvent control / 5% acetone, 2 µL | Treatment Cohort |
| **`10 % acetone 2 microliter`** | `10% Acetone 2 µL` | Solvent control / 10% acetone, 2 µL | Treatment Cohort |
| **`5 % acetone 5 microliter`** | `5% Acetone 5 µL` | Solvent control / 5% acetone, 5 µL | High-Volume Treatment Cohort |

*Blinded Trials*: During blinded stages when treatments have not yet been unblinded, cohorts remain labeled as `None / Blinded` and raw file identifiers are used.

---

## 3. Resurrected Bee Observation Correction Rule

### Biological Challenge
Honeybees in oral toxicology studies frequently enter reversible knockdown, chill-coma, or lethargy states where they lie on their backs with motionless legs. A human experimenter might record such a bee as Dead (`D`) on Day $k$. If temperature recovers or the chemical clears, the bee may stand up and resume active feeding (`A` or `B`) on Day $j > k$.

### Algorithmic Resolution
To prevent false mortality spikes followed by impossible negative mortality rates:
$$\text{Status}(i, d) = \begin{cases} \text{Alive}, & \text{if } \exists j \ge d \text{ such that observation is } A \text{ or } B \\ \text{Dead}, & \text{otherwise} \end{cases}$$
A bee is classified as deceased at time $d^*$ if and only if **all subsequent observations through the end of the experiment remain Dead (`D`)**.

