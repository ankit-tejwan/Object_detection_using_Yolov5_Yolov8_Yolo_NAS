#####################################################################################
# This script will download all Yolo segmentation  model 
#####################################################################################
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        results = self.model.predict(source=img.copy(), iou=0.1, retina_masks=True, conf=0.5)
        return results

def generate_color(class_id, max_classes=256):
    """
    Generate a unique color based on the class ID.
    The color is generated by varying the Hue value in the HSV color space.
    """
    hue = int((class_id * 179) / max_classes)  # Ensure unique hues for up to max_classes
    saturation = 255  # Maximum saturation
    value = 255  # Maximum brightness
    color = cv2.cvtColor(np.array([[[hue, saturation, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in color)


ys = YOLOSegmentation("yolov8s-seg.pt")

# Capture video from the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        # img dimensions
        img_height, img_width = img.shape[:2]
        # Print image dimensions
        print(f"Image dimensions: {img_height} x {img_width}")
        if not ret:
            break

        # Detect objects in the frame
        results = ys.detect(img)

        for result in results:
            masks = result.masks
            class_ids = result.boxes.cls.tolist()
            confidence = result.boxes.conf.tolist()

            if masks is not None:
                for i in range(len(masks)):
                    mask = masks[i].xy
                    class_id = class_ids[i]
                    color = generate_color(class_id)
                                        # Define the class names (this could be part of the YOLO model metadata, or you can define your own list)
                    classes = ys.model.names
                    for segment in mask:
                        segment = np.array(segment, dtype=np.int32)
                        # Draw filled polygon
                        #cv2.fillPoly(img, [segment.reshape(-1, 1, 2)], color)
                        bbox = cv2.boundingRect(segment)
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

                        # Draw class name
                        class_name = classes[int(class_id)] if int(class_id) < len(classes) else "Unknown"  # Cast class_id to int
                        label = f"{class_name} {confidence[i]:.2f}"
                        # print label
                        print(f"{label}")
                        cv2.putText(img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting image
        cv2.imshow("image", img)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
