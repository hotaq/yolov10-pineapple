import os
import argparse
import imgaug.augmenters as iaa
import imageio
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def create_red_and_pink_mask(image):
    """Create a mask that keeps both red and pink colors unchanged."""
    # Convert the image to HSV color space
    image_hsv = rgb2hsv(image.astype(np.float32) / 255.0)  # Convert to float in [0, 1] range

    # Hue values in the HSV color space
    hue = image_hsv[:, :, 0]

    # Define hue ranges for red and pink
    red_mask = (hue >= 0.0) & (hue <= 0.1)  # Red hue range
    pink_mask = (hue >= 0.9) & (hue <= 1.0)
    red_mask1 = (hue >= 0.0) & (hue <= 0.05)  # Pink hue range

    # Combine the masks
    red_and_pink_mask = red_mask | pink_mask | red_mask1

    return red_and_pink_mask

def main(input_folder, fixed_hue_value):
    # Define the augmenter
    aug = iaa.WithHueAndSaturation(
        iaa.WithChannels(0, iaa.Add(fixed_hue_value)),
        # Add a fixed value to the hue channel
    )
    aug1 = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
    aug1 = iaa.Grayscale(alpha=1.0)

    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image = imageio.imread(image_path)

            # Create a mask for red and pink colors
            red_mask = create_red_and_pink_mask(image)

            # Apply the augmentation
            augmented_image = aug1(image=image)

            # Apply the red mask to keep red and pink colors unchanged
            augmented_image[red_mask] = image[red_mask]

            # Save the augmented image to the output folder with the same name
            save_path = os.path.join(input_folder, filename)
            imageio.imwrite(save_path, augmented_image.astype(np.uint8))
            print(f"Processed and saved: {save_path}")

    print("All images have been processed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image augmentation script.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the directory where output images will be saved.")
    parser.add_argument('--fixed_hue_value', type=int, default=100, help="Fixed hue value to add for augmentation.")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.fixed_hue_value)
