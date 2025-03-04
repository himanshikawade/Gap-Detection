# List of class labels for COCO dataset (80 object categories)
labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", 
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
    "zebra", "giraffe", "N/A", "backpack", "umbrella", "handbag", "tie", "suitcase", 
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
    "bed", "dining table", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "book","cylindrical shape"
]

# Write the labels to a file
with open("coco.names", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print("coco.names file has been generated.")
