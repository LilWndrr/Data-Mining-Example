import os 
import csv
import numpy as np
import statistics
from scipy import stats

import pandas as pd
import seaborn as sns

# --- NEW IMPORTS HERE ---
import matplotlib.pyplot as plt
from collections import Counter
# ------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'my_location_features.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data_list=[]
    for row in reader:
        data_list.append(row)

# Extract columns
activities = [row['Activity'] for row in data_list]
distances = [float(row['Distance_m']) for row in data_list]
durations = [float(row['Duration_min']) for row in data_list]
speeds = [float(row['Speed_m_min']) for row in data_list]       
time_periods = [row['Time_Period'] for row in data_list]


# Calculate Stats
activitiesMode= statistics.mode(activities)

time_periodsMode= statistics.mode(time_periods)

distancesMeasurments= {
    'mean': np.mean(distances),
    'median': np.median(distances),
    'stdev': np.std(distances),
    'variance': np.var(distances),
    'min': np.min(distances),
    'max': np.max(distances)
}
durationsMeasurments= {
    'mean': np.mean(durations),
    'median': np.median(durations),
    'stdev': np.std(durations),
    'variance': np.var(durations),
    'min': np.min(durations),
    'max': np.max(durations)
}
speedsMeasurments= {
    'mean': np.mean(speeds),
    'median': np.median(speeds),
    'stdev': np.std(speeds),
    'variance': np.var(speeds),
    'min': np.min(speeds),
    'max': np.max(speeds)
}

# Print Stats
print("Distance measurements:", distancesMeasurments)
print("Mode of activities:", activitiesMode)
print("Mode of time periods:", time_periodsMode)
print("Duration measurements:", durationsMeasurments)
print("Speed measurements:", speedsMeasurments)

# --- PLACE THE NEW PLOTTING CODE HERE (AT THE END) ---

# 1. Count the activities
activity_counts = Counter(activities)

# 2. Print percentage breakdown
print("\n--- Activity Class Balance ---")
total_samples = len(activities)
for act, count in activity_counts.most_common():
    percentage = (count / total_samples) * 100
    print(f"{act}: {count} ({percentage:.1f}%)")

# 3. Draw the plot
plt.figure(figsize=(10, 6))
# Convert keys and values to lists explicitly
bars = plt.bar(list(activity_counts.keys()), list(activity_counts.values()), color='cornflowerblue', edgecolor='black')

plt.title('Class Balance: Samples per Activity')
plt.xlabel('Activity Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)

# Add counts on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()
# -----------------------------------------------------

df = pd.read_csv(csv_path)
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))

sns.boxplot(x='Activity', y='Speed_m_min', data=df, palette='Set2',showfliers=False)
plt.title('Speed Distribution by Activity Type')
plt.xlabel('Activity Type')
plt.ylabel('Speed (m/min)')
plt.tight_layout()
plt.show()

activity_hour=pd.crosstab(df['Time_Period'], df['Activity'])

activity_hour.plot(kind='bar', stacked=True, figsize=(12,6), colormap='viridis', width=0.8)
plt.title('Activity Count by Time Period')
plt.xlabel('Activity Type')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.show()

