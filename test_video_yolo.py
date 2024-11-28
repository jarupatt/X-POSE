import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('model_class.pt')

# Open the video file
video_path = 'test-video_18MdvkIA.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation using YOLOv8
    results = model(frame)

    # Iterate over the detected poses and draw keypoints if available
    if results[0].keypoints is not None:
        for result in results[0].keypoints:
            if result is not None:
                for keypoint in result:  # Each keypoint contains [x, y, confidence]
                    if len(keypoint) == 3:
                        x, y, conf = keypoint
                        if conf > 0.5:  # Only consider keypoints with high confidence
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoints

    # Display the class label on the top-left of the frame
    if results[0].boxes is not None:
        for det in results[0].boxes:
            # The bounding box data: [x1, y1, x2, y2, conf, cls]
            xyxy = det.xyxy[0]  # Bounding box coordinates: [x1, y1, x2, y2]
            conf = det.conf[0]  # Confidence score
            cls = det.cls[0]  # Class index

            # Extract the label
            label = f'{model.names[int(cls)]} {conf:.2f}'

            # Draw the bounding box
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

            # Draw the label on top of the bounding box
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
