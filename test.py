import pickle
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tempfile
import os

# Title
st.title("Pose Exercise")

# File upload section
st.header("Select Video File")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

# Process button
process_button = st.button("Process Video")

with open('exercise_nn_model_lat.pkl', 'rb') as f:
    modelnn = pickle.load(f)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_midpoint(coord1, coord2):
    x = (coord1[0] + coord2[0]) / 2
    y = (coord1[1] + coord2[1]) / 2
    z = (coord1[2] + coord2[2]) / 2
    return np.array([x, y, z])


def calculate_angle_using_cross_product(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cross_product = np.cross(ba, bc)
    dot_product = np.dot(ba, bc)
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
    return np.degrees(angle)


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Load YOLO model
model = YOLO('model_final.pt')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)


def process_video(uploaded_file):
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_file_path)

    if not cap.isOpened():
        st.error("Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    progress_bar = st.progress(0)
    frame_count = 0
    ex_elbo_dis = [0.0, 0.0, 0.0]

    lat_max = []
    squat_angle = 1.0
    squat_stand = 0.0
    elbow_angle_temp = 0.0
    puttext = 0
    error_el = 0

    # Define a temporary output path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_file:
        output_path = tmp_output_file.name

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Class mapping
    names = model.names

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Perform YOLO inference
            results = model(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)

            # Process YOLO class predictions
            if results and len(results) > 0 and results[0].probs is not None:
                top1 = results[0].probs.top1  # Top class index
                classname = names[top1]  # Class name from index


                cv2.putText(frame, f'Class: {classname}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if puttext != 0 and error_el != 0:
                    cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    puttext -= 1
                    if puttext == 0:
                        error_el = 0

                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = pose_results.pose_landmarks.landmark

                    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                    # Flatten landmarks for model input
                    pose_data = landmarks_array.flatten()

                    pose_data = pose_data.reshape(1, -1)

                    predicted_class = modelnn.predict(pose_data)[0]

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

                    # Process specific classes
                    if classname == "bicep":
                        if left_shoulder.visibility > 0.5:
                            h, w, _ = frame.shape
                            midpoint_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                            cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot

                            angle = calculate_angle(
                                [left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                [left_elbow.x, left_elbow.y, left_elbow.z],
                                [left_wrist.x, left_wrist.y, left_wrist.z]
                            )
                            cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 0), 2)

                            if frame_count > 3:
                                elbow_angle = calculate_angle([left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                                              [left_elbow.x, left_elbow.y, left_elbow.z],
                                                              [left_hip.x, left_hip.y, left_hip.z])

                                if abs(elbow_angle - elbow_angle_temp) > 35:
                                    error_el = 1
                                    puttext = 15  # 1 sec
                                    cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
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
                            cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 0), 2)

                            if frame_count > 3:
                                elbow_angle = calculate_angle([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                              [right_elbow.x, right_elbow.y, right_elbow.z],
                                                              [right_hip.x, right_hip.y, right_hip.z])

                                if abs(elbow_angle - elbow_angle_temp) > 35:
                                    error_el = 1
                                    puttext = 15  # 1 sec
                                    cv2.putText(frame, f'FIX Elbow', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                            if frame_count % 3 == 0:  # 2 sec
                                elbow_angle_temp = calculate_angle(
                                    [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                    [right_elbow.x, right_elbow.y, right_elbow.z],
                                    [right_hip.x, right_hip.y, right_hip.z])

                    elif classname == "dumbbell row":  # dumbbell row
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
                                cv2.putText(frame, f'Elbow down', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                            2)

                            if leftangle < np.degrees(1.5):  # 100 degree
                                cv2.putText(frame, f'keep left arm straight', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                        else:
                            back_mid = calculate_midpoint([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                          [right_hip.x, right_hip.y, right_hip.z])
                            if right_elbow.y * 1.05 < back_mid[1]:
                                cv2.putText(frame, f'Elbow down', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                            2)

                            if rightangle < np.degrees(1.5):
                                cv2.putText(frame, f'keep right arm straight', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                    elif classname == "incline dumbbell press":  # Incline Dumbbell Press
                        angle = calculate_angle(
                            [left_shoulder.x, left_shoulder.y],
                            [left_elbow.x, left_elbow.y],
                            [left_wrist.x, left_wrist.y]
                        )
                        cv2.putText(frame, f'Elbow Angle: {int(angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)

                        h, w, _ = frame.shape
                        midpoint_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                        cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot
                        midpoint_pixel = (int(right_elbow.x * w), int(right_elbow.y * h))
                        cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot
                        midpoint_pixel = (int(right_hip.x * w), int(right_hip.y * h))
                        cv2.circle(frame, midpoint_pixel, 5, (0, 255, 0), -1)  # Green dot

                        if left_elbow.y * 0.95 > left_shoulder.y:
                            shoulder_angle = calculate_angle(
                                [right_shoulder.x, right_shoulder.y],
                                [right_elbow.x, right_elbow.y],
                                [right_hip.x, right_hip.y]
                            )

                            cv2.putText(frame, f'shoulder_ang: {int(shoulder_angle)}', (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2)
                            if shoulder_angle > np.degrees(2.7925268):  # ~160 degrees
                                cv2.putText(frame, f'too wide elbow', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                    elif classname == "lat pulldown":  # Lat Pull Down
                        shoulder_distance = calculate_distance(
                            [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                            [left_shoulder.x, left_shoulder.y, left_shoulder.z]
                        )
                        wrist_distance = calculate_distance(
                            [right_wrist.x, right_wrist.y, right_wrist.z],
                            [left_wrist.x, left_wrist.y, left_wrist.z]
                        )

                        if wrist_distance > shoulder_distance * 2.2:
                            cv2.putText(frame, "Keep Wrists Closer", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)

                    elif classname == "squat":  # Squat (already implemented)
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
                                    cv2.putText(frame, f'adjust your ankle closer', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 0), 2)

                                if ankle_distance < (shoulder_distance * 0.8):
                                    error_sh_kne = 1
                                    cv2.putText(frame, f'adjust your ankle wider', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 0), 2)

                            if squat_stand != 0:
                                if right_knee.x < right_foot_index.x:
                                    if right_knee.x + squat_stand < right_foot_index.x:
                                        cv2.putText(frame, f'false position', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)

                        else:
                            if squat_angle > np.degrees(2.7925268):
                                squat_stand = abs(left_knee.x - left_foot_index.x)
                                if ankle_distance > (shoulder_distance * 1.43):
                                    error_sh_kne = 1
                                    cv2.putText(frame, f'adjust your ankle closer', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 0), 2)

                                if ankle_distance < (shoulder_distance * 0.8):
                                    error_sh_kne = 1
                                    cv2.putText(frame, f'adjust your ankle wider', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 0), 2)

                            if squat_stand != 0:
                                if left_knee.x < left_foot_index.x:

                                    if left_knee.x + squat_stand < left_foot_index.x:
                                        cv2.putText(frame, f'false position', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)

            progress_bar.progress(int(frame_count / total_frames * 100))

            # Write the frame to the output video
            out.write(frame)

    finally:
        cap.release()
        out.release()

        import time
        time.sleep(1)
        # Display the processed video
        st.success("Video processing is complete!")
        # st.video(output_path)

        # Provide download option
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )

        # Cleanup temporary files
        os.remove(tmp_file_path)
        os.remove(output_path)


if uploaded_file is not None and process_button:
    process_video(uploaded_file)
elif uploaded_file is None:
    st.error("Please upload a video file.")
