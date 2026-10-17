import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray

# 1. Load the sunflower field image
image_path = 'A02/a2images/the_berry_farms_sunflower_field.jpeg'
im = cv.imread(image_path, cv.IMREAD_REDUCED_COLOR_4)

if im is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
else:
    # Convert image to grayscale and normalize values to [0, 1]
    image_gray = rgb2gray(cv.cvtColor(im, cv.COLOR_BGR2RGB))

    # 2. Define Scale-Space Parameters for LoG Blob Detection
    min_sigma = 1
    max_sigma = 30
    num_sigma = 10
    threshold = 0.1  # Adjust sensitivity depending on image contrast

    print(f"--- Scale-Space Configuration ---")
    print(f"Minimum Sigma (min_sigma): {min_sigma}")
    print(f"Maximum Sigma (max_sigma): {max_sigma}")
    print(f"Number of Sigma steps (num_sigma): {num_sigma}")

    # Perform Laplacian of Gaussian (LoG) blob detection
    # Returns an array of coordinates and scales: [y, x, sigma]
    blobs_log = blob_log(
        image_gray, 
        min_sigma=min_sigma, 
        max_sigma=max_sigma, 
        num_sigma=num_sigma, 
        threshold=threshold
    )

    # Convert sigma to circle radius: r = sigma * sqrt(2)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

    # 3. Report parameters of the largest circles
    if len(blobs_log) > 0:
        # Sort blobs by radius in descending order
        sorted_blobs = blobs_log[np.argsort(blobs_log[:, 2])[::-1]]
        
        print("\n--- Top 5 Largest Detected Circles ---")
        for i in range(min(5, len(sorted_blobs))):
            y, x, r = sorted_blobs[i]
            print(f"Circle {i+1}: Center X = {x:.2f}, Center Y = {y:.2f}, Radius (r) = {r:.2f}")

        # 4. Visualize and draw circles on the image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        
        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='yellow', linewidth=1.2, fill=False)
            ax.add_patch(c)
            
        plt.title("Sunflower Field Blob Detection (Laplacian of Gaussians)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No blobs detected. Try lowering the threshold value.")