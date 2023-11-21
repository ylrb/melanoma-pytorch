import os
import pandas as pd

# Load the CSV file into a Pandas DataFrame
csv_path = "train.csv"
df = pd.read_csv(csv_path)

# Filter rows where the target is 0
filtered_df = df[df['target'] == 0]

# Get the list of image names to be deleted
images_to_delete = filtered_df['image_name'].tolist()

# Specify the path to the folder containing images
image_folder_path = "train"

# Delete images with target 0
for image_name in images_to_delete:
    image_path = os.path.join(image_folder_path, f"{image_name}.jpg")
    try:
        os.remove(image_path)
        print(f"Deleted: {image_path}")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except Exception as e:
        print(f"Error deleting {image_path}: {e}")

print("Deletion process complete.")
