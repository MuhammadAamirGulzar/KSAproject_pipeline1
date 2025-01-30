import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_image(image_path,count):
       #Output Folder
    output_folder=r'D:\CheckOutput'
    if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    # Load the image using OpenCV
    image = cv2.imread(image_path)
   
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Calculate the Laplacian variance of the grayscale image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
   
    # If variance is too low, indicating a uniform region, adjust the threshold
    if variance < 50:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    else:
        # Apply a binary threshold to get the detailed regions
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # Sort contours by area (descending) to get top 10 largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
   
    # Initialize an empty mask to draw the contours
    mask = np.zeros_like(gray)
   
    # Draw the top 10 contours on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
   
    # Find bounding box coordinates of the masked area
    x, y, w, h = cv2.boundingRect(mask)
   
    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    img_name=f'Output_Image_{image_num}.png'
    output=os.path.join(output_folder,img_name)
    cropped_img.save(output)
   
    return image, cropped_image

def calculate_reduction(original_image, cropped_image):
    original_height, original_width = original_image.shape[:2]
    cropped_height, cropped_width = cropped_image.shape[:2]
   
    original_area = original_height * original_width
    cropped_area = cropped_height * cropped_width
   
    reduction_percent = 100 * (original_area - cropped_area) / original_area
   
    return original_width, original_height, cropped_width, cropped_height, reduction_percent

def Image_Read():    
    folder_path = 'D:\check'
    count=0
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  
            filepath = os.path.join(folder_path, filename)
            original_image, cropped_image = crop_image(filename)
            original_width, original_height, cropped_width, cropped_height, reduction_percent = calculate_reduction(original_image, cropped_image)
            print(f"Original Image - Width: {original_width}, Height: {original_height}")
            print(f"Cropped Image - Width: {cropped_width}, Height: {cropped_height}")
            print(f"Reduction Percentage: {reduction_percent:.2f}%")
           
            count+=1
           
        else:
            print(f'Error reading file: {filename}')

def main():
    Image_Read()

main()
