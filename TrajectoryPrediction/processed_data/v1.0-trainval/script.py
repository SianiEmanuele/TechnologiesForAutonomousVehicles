import pandas as pd
import numpy as np

# Parametri limite
max_x = 100
max_y = 10

# Caricamento
df = pd.read_csv(r"TrajectoryPrediction\processed_data\v1.0-trainval\trajectories_dataset_copy.csv")

# Pulizia iniziale
df = df.drop(columns=["rot_q0", "rot_q1", "rot_q2", "rot_q3"])
df = df[df["category_2"] != 1]
df["speed"] = np.sqrt(df["v_x"]**2 + df["v_y"]**2)
df["heading"] = np.arctan2(df["v_y"], df["v_x"])

# Nuovo DataFrame per le istanze filtrate
filtered_rows = []
initial_instances = df["instance_token"].nunique()

# Raggruppamento per istanza
for instance_token, group in df.groupby("instance_token"):
    x_rel = group["x_rel"].values
    y_rel = group["y_rel"].values

    # Trova il primo indice che supera i limiti
    first_exceeding_x_index = next((i for i, x in enumerate(x_rel) if abs(x) > max_x), len(x_rel))
    first_exceeding_y_index = next((i for i, y in enumerate(y_rel) if abs(y) > max_y), len(y_rel))
    limit_index = min(first_exceeding_x_index, first_exceeding_y_index)

    if limit_index < 16:
        continue

    valid_group = group.iloc[:limit_index]
    filtered_rows.append(valid_group)

# Concatenazione
filtered_df = pd.concat(filtered_rows, ignore_index=True)
final_instances = filtered_df["instance_token"].nunique()

# Reordering and saving
filtered_df = filtered_df[[
    "instance_token", "x_rel", "y_rel", 
    "speed", "heading", "timestamp", 
    "category_0", "category_1"
]]
filtered_df.to_csv(r"TrajectoryPrediction\processed_data\v1.0-trainval\trajectories_dataset.csv", index=False)

# Report
print(f"Istanze iniziali: {initial_instances}")
print(f"Istanze finali: {final_instances}")
