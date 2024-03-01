import easyocr
import numpy as np
from PIL import Image
# import tensorflow as tf
# https://github.com/Foroozani/ImageProcessing
# Load your custom model (trained to detect 0 and 5)
# Replace with the actual path to your custom model
custom_model_path = "path/to/your/custom/model"
custom_model = tf.lo(custom_model_path)  # Assuming you've already loaded your custom model

# Load the input image
image_path = "C:/works/Persian_Arabic_OCR/test-1.jpg"
image = Image.open(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['fa', 'ar'], gpu=True)  # You can adjust the languages as needed

# Read text from the image
result = reader.readtext(np.array(image))  # Convert image to a numpy array

def extract_image_patch(image, position):
    """
    Extracts the region of interest (ROI) around the detected text position.

    Args:
        image (PIL.Image.Image): The original input image.
        position (tuple): (x, y) coordinates of the detected text.

    Returns:
        PIL.Image.Image: The extracted image patch (ROI).
    """
    x, y = position
    patch_size = 50  # Adjust the patch size based on your requirements
    roi = image.crop((x, y, x + patch_size, y + patch_size))
    return roi

# Process each detected text
for detection in result:
    detected_text, (x, y), _, _ = detection
    if "0" in detected_text or "5" in detected_text:
        # Extract the corresponding image patch (ROI) for prediction
        roi = extract_image_patch(image, (x, y))  # Implement this function

        # Predict using your custom model
        corrected_result = custom_model.predict(roi)
        print(f"Detected: {detected_text} | Corrected: {corrected_result}")
    else:
        print(f"Ignoring: {detected_text} (Not 0 or 5)")

# Implement the custom_model.predict function as needed
# Adjust the patch size and other parameters according to your use case
