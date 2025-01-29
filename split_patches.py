from PIL import Image
import os
from tqdm import tqdm

def split_image_into_patches(image_path, output_folder, overlap):
    # Output name
    output_name = image_path.split('\\')[-1][:len('You.Shou.Yan.0001')]

    # Open the image
    img = Image.open(image_path)
    width, height = img.size
    patch_size = width // 4

    # Calculate the step size (patch size minus overlap)
    step = int(patch_size * (1 - overlap))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate patches
    patch_id = 0
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            # Crop the patch
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            # Save the patch
            patch.save(os.path.join(output_folder, f"{output_name}_patch_{patch_id:02d}.png"))
            patch_id += 1

    # print(f"Generated {patch_id} patches in '{output_folder}'.")

# Parameters
folder_path = "database\You.Shou.Yan-comic-en"
output_folder = "database\You.Shou.Yan-comic-en-patches-266"
overlap = 0

for image_path in tqdm(os.listdir(folder_path)):
    image_path = os.path.join(folder_path, image_path)
    split_image_into_patches(image_path, output_folder, overlap)
