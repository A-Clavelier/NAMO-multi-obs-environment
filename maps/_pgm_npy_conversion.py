import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the PGM file in grayscale mode
image_array = cv2.imread("./maps/200x200_map2/SCmap.pgm", cv2.IMREAD_GRAYSCALE)
# image_array = np.load("./maps/500x500_map1/SCmap.npy")

# Check if the image was loaded correctly
if image_array is not None:
    
    image_array = image_array.astype(np.float32)
    image_array=(2*image_array/255)-1

    plt.imshow(image_array)
    plt.colorbar()
    plt.show()

    np.save('./maps/200x200_map2/SCmap.npy', image_array)
else:
    print("Error: Image not loaded. Check the file path and format.")