import cv2
import time
import torch

# Start timer
t1 = time.time()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')  # Detection model

# Load original frame
frame_path = 'cat.jpg'
original_frame = cv2.imread(frame_path)
if original_frame is None:
    raise FileNotFoundError(f"Image file {frame_path} not found.")

# Perform detection
results = model(original_frame)
detections = results.xyxy[0].cpu().numpy()  # Convert detections to NumPy array

# Draw bounding boxes on the original frame
detection_frame = original_frame.copy()  # Copy to avoid overwriting original_frame
for detection in detections:
    x1, y1, x2, y2, score, class_id = detection
    if score > 0.4:  # Apply confidence threshold
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        detection_frame = cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Format the score to show two decimal places
        formatted_score = f"{score:.2f}"
        # Draw the label and score
        label = f"{model.names[int(class_id)]} {formatted_score}"
        detection_frame = cv2.putText(detection_frame, label, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


# Save the result
cv2.imwrite('yolov5s_cat.jpg', detection_frame)
print("Image saved with detections: 'yolov5s_cat.jpg'")

# End timer and print total time taken
t2 = time.time()
time_taken = t2 - t1
print(f"Total time taken: {time_taken:.2f} seconds")
