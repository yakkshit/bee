import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, WeibullFitter, ExponentialFitter
from scipy.optimize import curve_fit

# Load data
df = pd.read_csv('observation_individual_bees.csv')
print(df)

# Separate bee observations and environmental metrics
bee_rows = df[df['bee_id'].str.isnumeric() | df['bee_id'].apply(lambda x: str(x).isdigit())]
env_rows = df[~df['bee_id'].isin(bee_rows['bee_id'])]

print("\nBee rows count:", len(bee_rows))
print("Env rows:\n", env_rows)

days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']

# For each bee, find event time and censoring
# A: Alive, D: Dead, B: Behavioral change (buzzed/abnormal, but alive)
durations = []
events = [] # 1 if dead, 0 if censored (survived through d6)

for idx, row in bee_rows.iterrows():
    b_id = row['bee_id']
    states = [row[d] for d in days]
    # Check if/when it died
    if 'D' in states:
        death_day = states.index('D') + 1 # 1-indexed day
        durations.append(death_day)
        events.append(1)
    else:
        durations.append(6) # censored at day 6
        events.append(0)

durations = np.array(durations)
events = np.array(events)

print("\nDurations:", durations)
print("Events:", events)

# Calculate daily counts
total_bees = len(bee_rows)
alive_counts = []
dead_counts = []
days_x = [0, 1, 2, 3, 4, 5, 6]

# Day 0: all alive
alive_counts.append(total_bees)
dead_counts.append(0)

for d_i, d in enumerate(days):
    d_num = d_i + 1
    # alive if duration > d_num or (duration == d_num and event == 0)
    # dead if duration <= d_num and event == 1
    num_dead = np.sum((durations <= d_num) & (events == 1))
    num_alive = total_bees - num_dead
    alive_counts.append(num_alive)
    dead_counts.append(num_dead)

survival_pct = np.array(alive_counts) / total_bees * 100
mortality_pct = np.array(dead_counts) / total_bees * 100

print("\nDays:", days_x)
print("Alive count:", alive_counts)
print("Dead count:", dead_counts)
print("Survival %:", np.round(survival_pct, 2))
print("Mortality %:", np.round(mortality_pct, 2))

# Kaplan Meier
kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events)
print("\nKM Median survival time:", kmf.median_survival_time_)
print("KM Survival Table:\n", kmf.survival_function_)
