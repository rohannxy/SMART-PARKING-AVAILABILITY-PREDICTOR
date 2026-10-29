# SMART PARKING AVAILABILITY PREDICTOR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create sample data
np.random.seed(42)
n = 300

data = pd.DataFrame({
    "Hour": np.random.randint(6, 23, n),
    "Weekend": np.random.randint(0, 2, n),
    "Event": np.random.randint(0, 2, n)
})

# 2. Generate simulated parking demand
demand = (
    20
    + 25 * data["Hour"].between(9, 11)
    + 25 * data["Hour"].between(13, 16)
    + 15 * (data["Weekend"] == 0)
    + 25 * data["Event"]
    + np.random.normal(0, 8, n)
)

# 3. Categorize parking availability
data["Availability"] = np.where(
    demand < 45, "High",
    np.where(demand < 70, "Medium", "Low")
)

# 4. Train the machine learning model
X = data[["Hour", "Weekend", "Event"]]
y = data["Availability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# 5. Prepare graph data
counts = data["Availability"].value_counts().reindex(
    ["High", "Medium", "Low"], fill_value=0
)

low_by_hour = (
    data.assign(Low=(data["Availability"] == "Low"))
    .groupby("Hour")["Low"].mean() * 100
)

# 6. Display all three graphs in ONE window
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Graph 1: Bar chart
counts.plot(kind="bar", ax=axes[0])
axes[0].set_title("Parking Availability")
axes[0].set_xlabel("Availability")
axes[0].set_ylabel("Number of Records")
axes[0].tick_params(axis="x", rotation=0)

# Graph 2: Line chart
axes[1].plot(low_by_hour.index, low_by_hour.values, marker="o")
axes[1].set_title("Low Availability by Hour")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Low Availability (%)")
axes[1].grid(True)

# Graph 3: Pie chart
counts.plot(kind="pie", autopct="%1.1f%%", ax=axes[2])
axes[2].set_title("Availability Percentage")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show(block=False)

# 7. Predict parking availability
print("\n--- PARKING PREDICTION ---")

try:
    hour = int(input("Enter hour (6 to 22): "))
    weekend = input("Is it a weekend? (yes/no): ").strip().lower()
    event = input("Is there a special event? (yes/no): ").strip().lower()

    if not 6 <= hour <= 22:
        print("Enter an hour from 6 to 22.")
    elif weekend not in ["yes", "no"] or event not in ["yes", "no"]:
        print("Please answer yes or no.")
    else:
        new_data = pd.DataFrame({
            "Hour": [hour],
            "Weekend": [1 if weekend == "yes" else 0],
            "Event": [1 if event == "yes" else 0]
        })

        result = model.predict(new_data)[0]
        print("Predicted Parking Availability:", result)

except ValueError:
    print("Please enter a valid hour.")