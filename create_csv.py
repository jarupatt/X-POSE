import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

def calculate_angle_2d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(file_path, class_name, df, desired_fps=20):
    cap = cv2.VideoCapture(file_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / desired_fps))  # Ensure frame_interval is at least 1

    pose = mp_pose.Pose()
    frame_data = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                try:
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    right_shoulder_angle = calculate_angle_2d(right_hip, right_shoulder, right_elbow)
                    right_elbow_angle = calculate_angle_2d(right_shoulder, right_elbow, right_wrist)
                    right_hip_angle = calculate_angle_2d(right_knee, right_hip, right_shoulder)
                    right_knee_angle = calculate_angle_2d(right_hip, right_knee, right_ankle)

                    left_shoulder_angle = calculate_angle_2d(left_hip, left_shoulder, left_elbow)
                    left_elbow_angle = calculate_angle_2d(left_shoulder, left_elbow, left_wrist)
                    left_hip_angle = calculate_angle_2d(left_knee, left_hip, left_shoulder)
                    left_knee_angle = calculate_angle_2d(left_hip, left_knee, left_ankle)

                except IndexError:
                    right_shoulder_angle = np.nan
                    right_elbow_angle = np.nan
                    right_hip_angle = np.nan
                    right_knee_angle = np.nan
                    left_shoulder_angle = np.nan
                    left_elbow_angle = np.nan
                    left_hip_angle = np.nan
                    left_knee_angle = np.nan
            else:
                right_shoulder_angle = np.nan
                right_elbow_angle = np.nan
                right_hip_angle = np.nan
                right_knee_angle = np.nan
                left_shoulder_angle = np.nan
                left_elbow_angle = np.nan
                left_hip_angle = np.nan
                left_knee_angle = np.nan

            frame_data.append({
                'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                'right_shoulder_angle' : right_shoulder_angle,
                'right_elbow_angle' : right_elbow_angle,
                'right_hip_angle' : right_hip_angle,
                'right_knee_angle' : right_knee_angle,
                'left_shoulder_angle' : left_shoulder_angle,
                'left_elbow_angle' : left_elbow_angle,
                'left_hip_angle' : left_hip_angle,
                'left_knee_angle' : left_knee_angle,
                'class_name': class_name
            })
        frame_count += 1

    cap.release()
    df = pd.concat([df, pd.DataFrame(frame_data)], ignore_index=True)
    return df

if __name__ == "__main__":
    video_folder = 'vid'  # Folder containing the videos
    output_folder = 'csv2'  # Folder to save CSV files

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize a single DataFrame to hold all data
    all_data = pd.DataFrame()

    # Process each video
    for file_name in os.listdir(video_folder):
        if file_name.endswith(('.mp4', '.MOV')):
            class_name = file_name.split('_')[0]  # Extract class name from file name
            file_path = os.path.join(video_folder, file_name)

            # Process the video and append data to the main DataFrame
            all_data = process_video(file_path, class_name, all_data)

    # Save the consolidated DataFrame to a single CSV file
    output_file = os.path.join(output_folder, 'output.csv')
    all_data.to_csv(output_file, index=False)
    print(f'Saved all angles to {output_file}')
