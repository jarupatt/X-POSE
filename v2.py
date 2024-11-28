from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import cv2
import pickle
from PIL import Image
import numpy as np

# Load the trained model
with open('./file/exercise_nn_model_lat.pkl', 'rb') as f:
    modelnn = pickle.load(f)

class Result:
    def __init__(self, names):
        self.names = names


def calculate_angle_between_three_points_2d(pointA, pointB, pointC):
    # Convert points to numpy arrays
    A = np.array(pointA)
    B = np.array(pointB)
    C = np.array(pointC)

    # Create vectors AB and BC
    AB = A - B
    BC = C - B

    # Normalize the vectors
    AB_norm = np.linalg.norm(AB)
    BC_norm = np.linalg.norm(BC)

    # Ensure the vectors are not zero-length to avoid division by zero
    if AB_norm == 0 or BC_norm == 0:
        return None

    AB_normalized = AB / AB_norm
    BC_normalized = BC / BC_norm

    # Calculate the dot product of the normalized vectors
    dot_product = np.dot(AB_normalized, BC_normalized)

    # Clip dot_product to avoid possible numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point

    # Calculate the vectors
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    return np.degrees(angle)


def calculate_midpoint(coord1, coord2):
    x = (coord1[0] + coord2[0]) / 2
    y = (coord1[1] + coord2[1]) / 2
    z = (coord1[2] + coord2[2]) / 2
    return np.array([x, y, z])


def calculate_angle_using_cross_product(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point

    # Calculate the vectors
    ba = a - b
    bc = c - b

    # Compute the cross product of the vectors
    cross_product = np.cross(ba, bc)

    # Compute the dot product
    dot_product = np.dot(ba, bc)

    # Calculate the angle using the cross and dot products
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)

    # Convert the angle from radians to degrees
    return np.degrees(angle)


# Function to calculate the distance between two points
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


# Load the model
model = YOLO('model_final.pt')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Open the video file
video_path = './file/er11_90.mp4'

cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
window_name = 'Frame'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 750)  # Resize the window to 800x600
# output_path = 'output_video.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
ex_elbo_dis = [0.0, 0.0, 0.0]

lat_max = []
squat_angle = 1.0
squat_stand = 0.0
elbow_angle_temp = 0.0
puttext = 0
error_el = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Perform inference on the current frame
    results = model(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose detection
    pose_results = pose.process(frame_rgb)

    names = results[0].names
    top1 = results[0].probs.top1
    if top1 == 0:
        classname = names[0]  # bicep
    if top1 == 1:
        classname = names[1]  # dumbbell row
    if top1 == 2:
        classname = names[2]  # incline dumbbell press
    if top1 == 3:
        classname = names[3]  # lat pull down
    if top1 == 4:
        classname = names[4]  # squat

    cv2.putText(frame, f'Class: {classname}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if puttext != 0 and error_el != 0:
        cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        puttext -= 1
        if puttext == 0:
            error_el = 0

    if pose_results.pose_landmarks:
        # Draw the skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Extract landmark positionst_ma
        landmarks = pose_results.pose_landmarks.landmark

        landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
        # Flatten landmarks for model input
        pose_data = landmarks_array.flatten()
        # Reshape data to match model's input format
        pose_data = pose_data.reshape(1, -1)
        # Predict the exercise class
        predicted_class = modelnn.predict(pose_data)[0]
        # Display the predicted class on the screen
        # cv2.putText(frame, f'Exercise: {predicted_class}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        if classname == names[0]:  # bicep
            if left_shoulder.visibility > 0.5:
                h, w, _ = frame.shape
                midpoint_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot

                angle = calculate_angle(
                    [left_shoulder.x, left_shoulder.y, left_shoulder.z],
                    [left_elbow.x, left_elbow.y, left_elbow.z],
                    [left_wrist.x, left_wrist.y, left_wrist.z]
                )
                cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if frame_count > 3:
                    elbow_angle = calculate_angle([left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                                  [left_elbow.x, left_elbow.y, left_elbow.z],
                                                  [left_hip.x, left_hip.y, left_hip.z])

                    if abs(elbow_angle - elbow_angle_temp) > 35:
                        error_el = 1
                        puttext = 15  # 1 sec
                        cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if frame_count % 3 == 0:  # 2 sec
                    elbow_angle_temp = calculate_angle([left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                                       [left_elbow.x, left_elbow.y, left_elbow.z],
                                                       [left_hip.x, left_hip.y, left_hip.z])
            else:
                h, w, _ = frame.shape
                midpoint_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot

                angle = calculate_angle(
                    [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                    [right_elbow.x, right_elbow.y, right_elbow.z],
                    [right_wrist.x, right_wrist.y, right_wrist.z]
                )
                cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if frame_count > 3:
                    elbow_angle = calculate_angle([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                  [right_elbow.x, right_elbow.y, right_elbow.z],
                                                  [right_hip.x, right_hip.y, right_hip.z])

                    if abs(elbow_angle - elbow_angle_temp) > 35:
                        error_el = 1
                        puttext = 15  # 1 sec
                        cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if frame_count % 3 == 0:  # 2 sec
                    elbow_angle_temp = calculate_angle([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                       [right_elbow.x, right_elbow.y, right_elbow.z],
                                                       [right_hip.x, right_hip.y, right_hip.z])

        elif classname == names[1]:  # dumbbell row
            leftangle = calculate_angle(
                [left_shoulder.x, left_shoulder.y, left_shoulder.z],
                [left_elbow.x, left_elbow.y, left_elbow.z],
                [left_wrist.x, left_wrist.y, left_wrist.z]
            )
            rightangle = calculate_angle(
                [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                [right_elbow.x, right_elbow.y, right_elbow.z],
                [right_wrist.x, right_wrist.y, right_wrist.z]
            )

            if leftangle > rightangle:
                back_mid = calculate_midpoint([left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                              [left_hip.x, left_hip.y, left_hip.z])

                if left_elbow.y * 1.05 < back_mid[1]:
                    cv2.putText(frame, f'Elbow down', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if leftangle < np.degrees(1.5):  # 100 degree
                    cv2.putText(frame, f'keep left arm straight', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
            else:
                back_mid = calculate_midpoint([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                              [right_hip.x, right_hip.y, right_hip.z])
                if right_elbow.y * 1.05 < back_mid[1]:
                    cv2.putText(frame, f'Elbow down', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if rightangle < np.degrees(1.5):
                    cv2.putText(frame, f'keep right arm straight', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)

        elif classname == names[2]:  # Incline Dumbbell Press
            angle = calculate_angle(
                [left_shoulder.x, left_shoulder.y, left_shoulder.z],
                [left_elbow.x, left_elbow.y, left_elbow.z],
                [left_wrist.x, left_wrist.y, left_wrist.z]
            )
            cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            h, w, _ = frame.shape
            midpoint_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot
            midpoint_pixel = (int(right_elbow.x * w), int(right_elbow.y * h))
            cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot
            midpoint_pixel = (int(right_hip.x * w), int(right_hip.y * h))
            cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot

            if left_elbow.y * 0.95 > left_shoulder.y:
                shoulder_angle = calculate_angle([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                 [right_elbow.x, right_elbow.y, right_elbow.z],
                                                 [right_hip.x, right_hip.y, right_hip.z])
                cv2.putText(frame, f'shoul_ang: {int(shoulder_angle)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

                if shoulder_angle > np.degrees(2.7925268):
                    cv2.putText(frame, f'too wide elbow', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif classname == names[3]:  # Lat Pull Down

            shoulder_distance = calculate_distance(
                [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                [left_shoulder.x, left_shoulder.y, left_shoulder.z]
            )
            wrist_distance = calculate_distance(
                [right_wrist.x, right_wrist.y, right_wrist.z],
                [left_wrist.x, left_wrist.y, left_wrist.z]
            )

            if wrist_distance > shoulder_distance * 2.2:
                cv2.putText(frame, "Keep Wrists Closer", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif classname == names[4]:  # Squat (already implemented)
            error_sh_kne = 0
            error_posi = 0

            shoulder_distance = calculate_distance(
                [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                [left_shoulder.x, left_shoulder.y, left_shoulder.z]
            )
            knee_distance = calculate_distance(
                [right_knee.x, right_knee.y, right_knee.z],
                [left_knee.x, left_knee.y, left_knee.z]
            )
            ankle_distance = calculate_distance(
                [right_ankle.x, right_ankle.y, right_ankle.z],
                [left_ankle.x, left_ankle.y, left_ankle.z]
            )

            squat_angle = calculate_angle(
                [right_hip.x, right_hip.y, right_hip.z],
                [right_knee.x, right_knee.y, right_knee.z],
                [right_ankle.x, right_ankle.y, right_ankle.z]
            )
            # flip =0
            if right_knee.visibility > 0.5:
                if squat_angle > np.degrees(2.7925268):
                    squat_stand = abs(right_knee.x - right_foot_index.x)
                    if ankle_distance > (shoulder_distance * 1.43):
                        error_sh_kne = 1
                        cv2.putText(frame, f'adjust your ankle closer', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                    if ankle_distance < (shoulder_distance * 0.8):
                        error_sh_kne = 1
                        cv2.putText(frame, f'adjust your ankle wider', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                if squat_stand != 0:
                    if right_knee.x < right_foot_index.x:
                        if right_knee.x + (squat_stand) < right_foot_index.x:
                            cv2.putText(frame, f'false position', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

            else:
                if squat_angle > np.degrees(2.7925268):
                    squat_stand = abs(left_knee.x - left_foot_index.x)
                    if ankle_distance > (shoulder_distance * 1.43):
                        error_sh_kne = 1
                        cv2.putText(frame, f'adjust your ankle closer', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                    if ankle_distance < (shoulder_distance * 0.8):
                        error_sh_kne = 1
                        cv2.putText(frame, f'adjust your ankle wider', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                if squat_stand != 0:
                    if left_knee.x < left_foot_index.x:

                        if left_knee.x + (squat_stand) < left_foot_index.x:
                            cv2.putText(frame, f'false position', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

    # Display the frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
# out.release()
cv2.destroyAllWindows()
