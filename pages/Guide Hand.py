import os
from PIL import Image
import streamlit as st

def display_image_grid(image_folder, rows, columns, image_width=200):
    st.title("Arabic Sign Language")

    st.write(
        "This page contains all kinds of Arabic sign language so any audience can understand and verify the validation detection hand")

    # List all image files in the specified folder
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Check if there are enough images for the grid
    if len(image_paths) < rows * columns:
        st.warning("Not enough images in the folder.")
        return

    # Create rows and columns
    image_index = 0
    for row in range(rows):
        col1, col2, col3 = st.columns(3)
        for col in [col1, col2, col3]:
            if image_index < len(image_paths):
                image_path = image_paths[image_index]
                image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract base name without extension
                image = Image.open(image_path)
                col.image(image, caption=image_name, width=image_width, channels="BGR")
                image_index += 1

# Specify the folder containing images
image_folder = "pages/assets/"

# Set the number of rows and columns
rows = 10
columns = 3

# Display the image grid
display_image_grid(image_folder, rows, columns)
