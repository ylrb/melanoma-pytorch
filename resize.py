import os
import cv2

# Specify the input and output directories
input_folder = "./malignant/"
output_folder = "./244-malignant/"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Specify the target size
target_size = (224, 224)
# Get the list of image files in the input folder

image_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".jpg")]
total_images = len(image_files)

print(total_images)

# Loop through each file in the input folder
for idx, filename in enumerate(image_files, start=1):
    # Read the image using OpenCV
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # Get the center crop
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    center_cropped = img[start_h:start_h + min_dim, start_w:start_w + min_dim, :]

    # Resize the image to the target size
    resized_img = cv2.resize(center_cropped, target_size)

    # Save the resized image to the output folder
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_img)

    # Print the percentage of the dataset processed
    percentage_processed = (idx / total_images) * 100
    print(f"Processed: {percentage_processed:.2f}% ({idx}/{total_images})")

print("Dataset creation complete.")

