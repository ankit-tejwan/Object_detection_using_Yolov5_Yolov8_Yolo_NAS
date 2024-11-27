import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Function to check the device (CPU or GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load a COCO-pretrained YOLOv8 model
model = YOLO("yolov8s.pt").to(device)

# Load the input image
input_image_path = 'cat.jpg'
image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Image not found or unable to load: {input_image_path}")

# Run inference with the YOLO model
results = model(input_image_path)

# Print results (for debugging purposes)
#print("Results:", results)

# Check if results contain detections
if not results or not results[0].boxes:
    print("No results returned from the model.")
    exit()

# Function to save the detected results with annotations
def save_detected_image(image, results, output_path, device):
    # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis for Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Process the first detection result (usually only one image input)
    boxes = results[0].boxes.xyxy.to(device).cpu().numpy()  # Get bounding box coordinates
    confidences = results[0].boxes.conf.to(device).cpu().numpy()  # Get confidence scores
    class_ids = results[0].boxes.cls.to(device).cpu().numpy()  # Get class ids

    # Loop over the detected boxes and plot them
    for box, conf, cls in zip(boxes, confidences, class_ids):
        xmin, ymin, xmax, ymax = box
        class_name = results[0].names[int(cls)]  # Get the class name

        # Draw the bounding box and label
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin, ymin, f'{class_name} {conf:.2f}', bbox={'facecolor': 'yellow', 'alpha': 0.5}, fontsize=12, color='black')

    # Remove axis
    ax.axis('off')

    # Save the output image with annotations
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

# Define output path for the annotated image
output_image_path = 'yolov8_cat_detected.jpg'

# Save the image with annotations
save_detected_image(image, results, output_image_path, device)

print(f"Annotated image saved to {output_image_path}")
