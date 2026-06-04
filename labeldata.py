import pandas as pd

data = pd.read_csv("sensor_data.csv", names=["timestamp","gas","water"])

def label(row):

    if row["gas"] > 400 and row["water"] == 1:
        return "danger"

    elif row["gas"] > 400:
        return "gas_leak"

    elif row["water"] == 1:
        return "water_high"

    else:
        return "normal"

data["label"] = data.apply(label, axis=1)

data.to_csv("labeled_data.csv", index=False)

print("Dataset labeled successfully")
