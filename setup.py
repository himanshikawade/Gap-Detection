import cv2
import numpy as np
from ultralytics import YOLO

def pixel_to_cm(pixel_distance, reference_pixel_height, reference_cm=20):
    """Converts pixel distance to cm using a known reference object height."""
    if reference_pixel_height == 0:
        return 0  # Avoid division by zero
    scale_factor = reference_cm / reference_pixel_height
    return pixel_distance * scale_factor

def main():
    model = YOLO("yolov8l.pt")  # Use a larger, more accurate model
    cap = cv2.VideoCapture(0)  # Open live camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)  # Increase confidence threshold
        bottle_box, book_box = None, None

        for result in results:
            for box in result.boxes:
                x, y, w, h = map(int, box.xywh[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                if label == "bottle":
                    bottle_box = (x, y, w, h)
                elif label == "book":
                    book_box = (x, y, w, h)
                
                # Draw bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if bottle_box and book_box:
            _, bottle_y, _, bottle_h = bottle_box
            _, book_y, _, book_h = book_box
            
            book_bottom = book_y + book_h
            bottle_top = bottle_y
            
            if book_bottom >= bottle_top:
                text = "No gap: 0 cm"
            else:
                pixel_gap = bottle_top - book_bottom
                real_gap_cm = pixel_to_cm(pixel_gap, bottle_h)
                text = f"Gap: {real_gap_cm:.2f} cm"
            
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Live Feed - Book & Bottle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()