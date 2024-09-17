from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load a COCO-pretrained YOLO-NAS-s model
model = YOLO("yolov8s.pt")

# Load the input image
input_image_path = 'cat.jpg'
image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Image not found or unable to load: {input_image_path}")

# Run inference with the YOLO-NAS-s model
results = model(input_image_path)

# Check if results contain detections
if not results:
    print("No results returned from the model.")
    exit()

# Function to save the detected results with annotations
def save_detected_image(image, results, output_path):
    # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis for Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Process results
    for result in results:
        # Get boxes from the result
        boxes = result.boxes.xyxy.numpy()  # Convert to numpy array
        confidences = result.boxes.conf.numpy()  # Convert to numpy array
        class_ids = result.boxes.cls.numpy()  # Convert to numpy array

        # Plot bounding boxes
        for box, conf, cls in zip(boxes, confidences, class_ids):
            xmin, ymin, xmax, ymax = box
            class_name = result.names[int(cls)]

            # Draw rectangle and label
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            plt.text(xmin, ymin, f'{class_name} {conf:.2f}', bbox={'facecolor': 'yellow', 'alpha': 0.5}, fontsize=12, color='black')

    # Remove axis
    ax.axis('off')

    # Save the output image
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

# Define output path
output_image_path = 'yolov8_cat_detected.jpg'

# Save the image with annotations
save_detected_image(image, results, output_image_path)

print(f"Annotated image saved to {output_image_path}")
