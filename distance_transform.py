import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded properly
    if img is None:
        raise Exception("Image not loaded. Check the path.")

    # Convert to binary image which the image is already in 
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return binary_img 


def compute_distance_transform(binary_img):
    # Invert image for distance transform (DT expects background to be white)
    inverted_img = 255 - binary_img
    # background should be white and placenta is black
    # 255 is the value for a white pixel
    # by subtracting the binary image from 255, the image is now going to have a black placenta and white background
    # Compute distance transform
    dist_transform = cv2.distanceTransform(inverted_img, cv2.DIST_L2, 5)
    # computes the distance to the closest zero pixel (closest white!! this is the problem currently) for each pixel of the image
    # Euclidean distance is used
    return dist_transform

def main():
    image_path = '/Users/ht/Downloads/placentaimage.png'  # Update the path
    binary_img = load_and_process_image(image_path)
    dist_transform = compute_distance_transform(binary_img)

    # Display results
    plt.imshow(dist_transform, cmap='jet')
    plt.title('Distance Transform')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
