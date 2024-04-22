import cv2
import numpy as np

def load_and_process_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded properly
    if img is None:
        raise Exception("Image not loaded. Check the path.")

    # Convert to binary image
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return binary_img

def compute_distance_transform(binary_img):
    # Invert the binary image to make the background black
    inverted_img = cv2.bitwise_not(binary_img)

    # Compute distance transform from the background
    dist_transform = cv2.distanceTransform(inverted_img, cv2.DIST_L2, 5)

    return dist_transform

def main():
    image_path = '/Users/ht/Downloads/placentaimage.png'  # Update the path
    binary_img = load_and_process_image(image_path)
    dist_transform = compute_distance_transform(binary_img)

    # Apply colormap to the distance map
    dist_transform_color = cv2.applyColorMap(np.uint8(dist_transform), cv2.COLORMAP_JET)

    # Display distance map with colormap
    cv2.imshow('Distance Map', dist_transform_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()