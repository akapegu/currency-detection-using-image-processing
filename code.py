"""
Currency Detection Using Image Processing
========================================

This script implements a computer vision-based currency detection system that can
identify and classify Indian Rupee and Japanese Yen banknotes using advanced
image processing techniques including custom Canny edge detection.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def closest(dataset, K):
    """
    Find the closest value from the dataset to the given value K.
    
    Args:
        dataset (list): List of reference values to compare against
        K (float): Target value to find the closest match for
    
    Returns:
        float: The closest value from the dataset
    """
    dataset = np.asarray(dataset) 
    idx = (np.abs(dataset - K)).argmin() 
    return dataset[idx]


def gray_gaussian_SobelFilter(image):
    """
    Convert image to grayscale, apply Gaussian blur, and then Sobel filtering.
    
    This function implements the first step of Canny edge detection:
    1. Convert BGR image to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply Sobel operators to compute gradients
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        tuple: (convolved_image, angles)
            - convolved_image: Edge magnitude image
            - angles: Gradient direction angles in degrees
    """
    # Convert BGR to grayscale for edge detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise (3x3 kernel, sigma=0)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Initialize arrays for gradient calculations
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)  # Gradient in x-direction
    G_y = np.zeros(image.shape)  # Gradient in y-direction
    size = image.shape
    
    # Sobel operators for gradient calculation
    x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))  # X-direction Sobel
    y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))  # Y-direction Sobel
    
    # Apply Sobel operators to compute gradients
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            # Compute gradients using 3x3 neighborhood
            G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], y))
    
    # Compute gradient magnitude
    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    # Compute gradient direction angles
    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180  # Normalize angles to 0-180 range
    convolved = convolved.astype('uint8')
    
    return convolved, angles


def non_maximum_suppression(image, angles):
    """
    Apply non-maximum suppression to thin edges.
    
    This function suppresses non-maximum pixels along the gradient direction,
    keeping only the strongest edge pixels. This is a key step in Canny edge detection.
    
    Args:
        image (numpy.ndarray): Gradient magnitude image
        angles (numpy.ndarray): Gradient direction angles
    
    Returns:
        numpy.ndarray: Thinned edge image after non-maximum suppression
    """
    size = image.shape
    suppressed = np.zeros(size)
    
    # Process each pixel (excluding borders)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            # Determine comparison pixels based on gradient angle
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                # Horizontal edge: compare with left and right pixels
                compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                # Diagonal edge (top-left to bottom-right): compare with diagonal pixels
                compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                # Vertical edge: compare with top and bottom pixels
                compare = max(image[i - 1, j], image[i + 1, j])
            else:
                # Diagonal edge (top-right to bottom-left): compare with diagonal pixels
                compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            # Keep pixel if it's the maximum in its gradient direction
            if image[i, j] >= compare:
                suppressed[i, j] = image[i, j]
    
    # Normalize the output
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

    
def double_threshold_hysteresis(image, low, high):
    """
    Apply double thresholding and hysteresis for edge linking.
    
    This function implements the final step of Canny edge detection:
    1. Apply two thresholds to classify pixels as strong, weak, or non-edges
    2. Use hysteresis to link weak edges to strong edges
    
    Args:
        image (numpy.ndarray): Input image after non-maximum suppression
        low (int): Low threshold for weak edges
        high (int): High threshold for strong edges
    
    Returns:
        numpy.ndarray: Final binary edge image
    """
    weak = 50   # Value for weak edge pixels
    strong = 250  # Value for strong edge pixels
    size = image.shape
    result = np.zeros(size)
    
    # Find weak and strong edge pixels
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    
    # Mark strong and weak pixels
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    
    # Define 8-connectivity neighborhood for edge linking
    dx = np.array((-1, 0, 1, -2, 0, 2, -1, 0, 1))
    dy = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1))
    
    # Hysteresis: link weak edges to strong edges
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        
        # Check 8-connectivity neighborhood
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            
            # Check if neighbor is within bounds and is a weak edge
            if ((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and 
                (result[new_x, new_y] == weak)):
                # Convert weak edge to strong edge
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    
    # Remove all non-strong edges
    result[result != strong] = 0
    return result


def Canny(image, low, high):
    """
    Complete Canny edge detection implementation.
    
    This function combines all steps of the Canny edge detection algorithm:
    1. Grayscale conversion and Gaussian blur
    2. Sobel filtering for gradient computation
    3. Non-maximum suppression for edge thinning
    4. Double thresholding and hysteresis for edge linking
    
    Args:
        image (numpy.ndarray): Input BGR image
        low (int): Low threshold for edge detection
        high (int): High threshold for edge detection
    
    Returns:
        numpy.ndarray: Binary edge image
    """
    image, angles = gray_gaussian_SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    image = double_threshold_hysteresis(image, low, high)
    return image


def checkforcountry(image):
    """
    Determine the currency type (Indian Rupee or Japanese Yen) from the image.
    
    This function analyzes a specific region of the edge-detected image to
    determine whether the currency is Indian Rupee or Japanese Yen based on
    the presence of bright pixels in a characteristic region.
    
    Args:
        image (numpy.ndarray): Edge-detected image
    
    Returns:
        tuple: (count, display_image)
            - count: Number of bright pixels (0 for INR, >0 for JPY)
            - display_image: Cropped region for visualization
    """
    # Define cropping region for currency classification
    # This region is empirically determined to distinguish between INR and JPY
    x1 = int(len(image) * 0.285)   # 28.5% of image height
    x2 = int(len(image) * 0.5714)  # 57.14% of image height
    y1 = int(len(image[0]) * 0.75) # 75% of image width
    y2 = int(len(image[0]) * 0.86) # 86% of image width
    
    display = []
    c0 = 0  # Counter for bright pixels
    temp = image[x1:x2]

    # Analyze the cropped region
    for i in range(len(temp)):
        temp1 = list(temp[i])
        temp2 = temp1[y1:y2]

        # Count bright pixels (>100 intensity)
        for i in range(len(temp2)):
            if temp2[i] > 100:
                c0 += 1

        temp3 = np.array(temp2)
        display.append(temp3)
    
    display = np.array(display)
    final_out = cv2.resize(display, (0, 0), fx=2.75, fy=2.75)

    return c0, final_out


def yenvaluedetection(image):
    """
    Detect the specific value of Japanese Yen banknotes.
    
    This function analyzes a specific region of the image to determine
    the denomination of Japanese Yen (1000, 2000, 5000, or 10000).
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        tuple: (index, extracted_image, cannied)
            - index: Index corresponding to the detected value
            - extracted_image: Cropped region for visualization
            - cannied: Edge-detected image
    """
    # Pre-defined dataset of dark/bright pixel ratios for different Yen values
    # These values are empirically determined from training data
    dataset_for_yen = [6.828, 7.027, 6.8936, 5.4565]  # Corresponds to [1000, 2000, 5000, 10000] Yen

    # Apply Canny edge detection
    cannied = Canny(image, 20, 80)
    x = cannied
    
    # Define cropping region for Yen value detection
    # This region contains characteristic features for Yen denomination
    x1 = 0
    x2 = int(len(x) * 0.2)  # Top 20% of image height
    y1 = int(len(x[0]) * 0.8336)  # 83.36% of image width
    y2 = int(len(x[0]) * 0.98)    # 98% of image width

    out = []
    xx = x[x1:x2]

    c0 = 0  # Counter for dark pixels (0 intensity)
    c1 = 0  # Counter for bright pixels (>0 intensity)
    
    # Analyze the cropped region
    for i in range(len(xx)):
        temp = list(xx[i])
        temp1 = temp[y1:y2]

        # Count dark and bright pixels
        for i in range(len(temp1)):
            if temp1[i] == 0:
                c0 += 1
            else:
                c1 += 1
        temp2 = np.array(temp1)
        out.append(temp2)
    
    out = np.array(out)
    final_outt = cv2.resize(out, (0, 0), fx=2.75, fy=2.75)

    # Find the closest match in the dataset
    check = closest(dataset_for_yen, c0/c1)
    tempy = dataset_for_yen.index(check)

    return tempy, final_outt, cannied


def rupeevaluedetection(image):
    """
    Detect the specific value of Indian Rupee banknotes.
    
    This function analyzes a specific region of the image to determine
    the denomination of Indian Rupee (10, 50, 100, 200, 500, or 2000).
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        tuple: (index, extracted_image, cannied)
            - index: Index corresponding to the detected value
            - extracted_image: Cropped region for visualization
            - cannied: Edge-detected image
    """
    # Pre-defined dataset of dark/bright pixel ratios for different Rupee values
    # These values are empirically determined from training data
    dataset_for_rupee = [8.946, 8.242, 5.36, 9.2, 7.398, 7.436]  # Corresponds to [10, 50, 100, 200, 500, 2000] Rupees

    # Apply Canny edge detection with different parameters for Rupee
    cannied = cv2.Canny(image, 0, 180)
    x = cannied
    
    # Define cropping region for Rupee value detection
    # This region contains characteristic features for Rupee denomination
    x1 = int(len(x) * 0.6164)  # 61.64% of image height
    x2 = int(len(x) * 0.822)   # 82.2% of image height
    y1 = int(len(x[0]) * 0.74) # 74% of image width
    y2 = int(len(x[0]) * 0.91) # 91% of image width
    
    out = []
    xx = x[x1:x2]

    c0 = 0  # Counter for dark pixels (0 intensity)
    c1 = 0  # Counter for bright pixels (>0 intensity)
    
    # Analyze the cropped region
    for i in range(len(xx)):
        temp = list(xx[i])
        temp1 = temp[y1:y2]

        # Count dark and bright pixels
        for i in range(len(temp1)):
            if temp1[i] == 0:
                c0 += 1
            else:
                c1 += 1
        temp2 = np.array(temp1)
        out.append(temp2)
    
    out = np.array(out)
    final_outt = cv2.resize(out, (0, 0), fx=2.75, fy=2.75)

    # Find the closest match in the dataset
    check = closest(dataset_for_rupee, c0/c1)
    tempy = dataset_for_rupee.index(check)

    return tempy, final_outt, cannied


def show_yen(index):
    """
    Display the detected Japanese Yen value image.
    
    Args:
        index (int): Index corresponding to the detected Yen value
                    0: 1000 Yen, 1: 2000 Yen, 2: 5000 Yen, 3: 10000 Yen
    
    Returns:
        numpy.ndarray: Display image for the detected Yen value
    """
    # Load display images for different Yen values
    val1 = cv2.imread("images/display1000yen.jpg")
    val2 = cv2.imread("images/display2000yen.jpg")
    val3 = cv2.imread("images/display5000yen.jpg")
    val4 = cv2.imread("images/display10000yen.jpg")
    
    # Return the appropriate display image based on index
    if index == 0:
        imvalue = val1
    elif index == 1:
        imvalue = val2
    elif index == 2:
        imvalue = val3
    elif index == 3:
        imvalue = val4

    return imvalue


def show_rupee(index):
    """
    Display the detected Indian Rupee value image.
    
    Args:
        index (int): Index corresponding to the detected Rupee value
                    0: 10 Rupees, 1: 50 Rupees, 2: 100 Rupees,
                    3: 200 Rupees, 4: 500 Rupees, 5: 2000 Rupees
    
    Returns:
        numpy.ndarray: Display image for the detected Rupee value
    """
    # Load display images for different Rupee values
    val1 = cv2.imread("images/display10rupee.jpg")
    val2 = cv2.imread("images/display50rupee.jpg")
    val3 = cv2.imread("images/display100rupee.jpg")
    val4 = cv2.imread("images/display200rupee.jpg")
    val5 = cv2.imread("images/display500rupee.jpg")
    val6 = cv2.imread("images/display2000rupee.jpg")

    # Return the appropriate display image based on index
    if index == 0:
        imvalue = val1
    elif index == 1:
        imvalue = val2
    elif index == 2:
        imvalue = val3
    elif index == 3:
        imvalue = val4
    elif index == 4:
        imvalue = val5
    else:
        imvalue = val6

    return imvalue


# Main execution block
if __name__ == "__main__":
    # Load and preprocess the input image
    # Note: Update the image path to point to your currency image in the images folder
    image = cv2.imread("images/5000yen.jfif")  # Example: testing with 5000 Yen image
    
    # Resize image to standard dimensions for processing
    image = cv2.resize(image, (600, int(600 * 0.45)))
    
    # Apply edge detection for currency classification
    detectcountry = cv2.Canny(image, 0, 180) 

    # Determine the currency type (INR or JPY)
    check, data_image = checkforcountry(detectcountry)

    # Process based on detected currency type
    if check == 0:
        # Indian Rupee detected
        print("Detected: Indian Rupee")
        indexx, extracted_image, canniedd = rupeevaluedetection(image)
        detected_value = show_rupee(indexx)
        
        # Resize images for display
        data_image = cv2.resize(data_image, (250, 270))
        extracted_image = cv2.resize(extracted_image, (250, 270))
        detected_value = cv2.resize(detected_value, (500, 270))

        # Create display layouts
        one = np.concatenate((image, detected_value), axis=1)  # Original + detected value
        two = np.concatenate((data_image, canniedd, extracted_image), axis=1)  # Processing steps

        # Display results
        cv2.imshow("Table 1 - Processing Steps", two)
        cv2.imshow("Table 2 - Original + Detected Value", one)
        cv2.waitKey(0)

    else:
        # Japanese Yen detected
        print("Detected: Japanese Yen")
        indexx, extracted_image, canniedd = yenvaluedetection(image)
        detected_value = show_yen(indexx)
        
        # Resize images for display
        data_image = cv2.resize(data_image, (250, 270))
        extracted_image = cv2.resize(extracted_image, (250, 270))
        detected_value = cv2.resize(detected_value, (500, 270))

        # Create display layouts
        one = np.concatenate((image, detected_value), axis=1)  # Original + detected value
        two = np.concatenate((data_image, canniedd, extracted_image), axis=1)  # Processing steps

        # Display results
        cv2.imshow("Table 1 - Processing Steps", two)
        cv2.imshow("Table 2 - Original + Detected Value", one)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
