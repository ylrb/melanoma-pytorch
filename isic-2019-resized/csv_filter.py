import pandas as pd

# Load the CSV file into a Pandas DataFrame
csv_path = "train.csv"
df = pd.read_csv(csv_path)

# Filter rows where the target is not equal to 0
filtered_df = df[df['target'] != 0]

# Save the filtered DataFrame back to a CSV file
filtered_df.to_csv("filtered_train.csv", index=False)

print("Rows with target 0 removed.")
