# -*- coding: utf-8 -*-
"""LMD metric.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qn9ydkl2hEjpnrN61HzUc560dfvh-kyE
"""

import numpy as np
import dlib
import cv2

def calculate_lmd(video_ref_path, video_synth_path, predictor_path):
    """
    Calculate the Landmark Distance (LMD) between the reference video and the synthesized video.

    Parameters:
    - video_ref_path: Path to the reference video.
    - video_synth_path: Path to the synthesized video.
    - predictor_path: Path to the shape predictor `.dat` file (e.g., shape_predictor_68_face_landmarks.dat).

    Returns:
    - lmd: The normalized Landmark Distance.
    """
    # Load Dlib's facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Initialize video capture objects
    cap_ref = cv2.VideoCapture(video_ref_path)
    cap_synth = cv2.VideoCapture(video_synth_path)

    lmd_total = 0.0
    frame_count = 0

    while True:
        # Read frames from both videos
        ret_ref, frame_ref = cap_ref.read()
        ret_synth, frame_synth = cap_synth.read()

        if not ret_ref or not ret_synth:
            break  # Exit if any video ends

        frame_count += 1

        # Convert frames to grayscale
        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        gray_synth = cv2.cvtColor(frame_synth, cv2.COLOR_BGR2GRAY)

        # Detect faces in both frames
        faces_ref = detector(gray_ref)
        faces_synth = detector(gray_synth)

        # Ensure both frames contain faces
        if len(faces_ref) > 0 and len(faces_synth) > 0:
            landmarks_ref = predictor(gray_ref, faces_ref[0])
            landmarks_synth = predictor(gray_synth, faces_synth[0])

            # Extract lip landmarks (points 48-67 in the 68-point model)
            points_ref = np.array([[p.x, p.y] for p in landmarks_ref.parts()[48:]])
            points_synth = np.array([[p.x, p.y] for p in landmarks_synth.parts()[48:]])

            # Calibrate the mean points of lip landmarks
            mean_ref = np.mean(points_ref, axis=0)
            mean_synth = np.mean(points_synth, axis=0)
            calibrated_ref = points_ref - mean_ref
            calibrated_synth = points_synth - mean_synth

            # Calculate the Euclidean distance between corresponding landmarks
            distances = np.linalg.norm(calibrated_ref - calibrated_synth, axis=1)
            lmd_total += np.sum(distances)

    # Normalizing the LMD by temporal length and number of landmark points
    T = frame_count
    P = len(points_ref)  # Total number of landmark points
    lmd = lmd_total / (T * P)

    # Release video capture objects
    cap_ref.release()
    cap_synth.release()

    return lmd

# Example usage
video_ref_path = "/content/drive/MyDrive/lipsync_metric/original.mp4"
video_synth_path = "/content/drive/MyDrive/lipsync_metric/model.mp4"
predictor_path = "/content/drive/MyDrive/lipsync_metric/shape_predictor_68_face_landmarks.dat"  # Dlib's pre-trained model

lmd = calculate_lmd(video_ref_path, video_synth_path, predictor_path)
print(f"Landmark Distance (LMD): {lmd}")