import cv2
import numpy as np
import librosa
import mediapipe as mp
from scipy.spatial import distance
from moviepy.editor import VideoFileClip

# Initialize mediapipe 
# face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_lip_features(frame):
    """
    Extracts the lip region features (x, y) from a given frame using MediaPipe.
    Returns the points corresponding to the lip area.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get face landmarks
    results = face_mesh.process(frame_rgb)
    
    lip_points = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        # MediaPipe lip indices (upper and lower lips)
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
        
        # Extract coordinates
        h, w, _ = frame.shape
        for idx in lip_indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            lip_points.append((x, y))
    
    return np.array(lip_points)

def compute_lip_sync_quality(lip_points, audio_features, fps=30):
    """
    Compute the quality of visual speech (lip sync quality) by comparing lip movement features
    with audio features.
    """
    if len(lip_points) < 2 or len(audio_features) < 2:
        return 0  # Return 0 if insufficient data
    
    # Calculate the Euclidean distance between consecutive lip points
    lip_distances = np.linalg.norm(np.diff(lip_points, axis=0), axis=1)
    
    # Modify this part to handle 1D audio features
    audio_distance = np.diff(audio_features)  # Remove axis parameter since audio_features is 1D
    
    # Compute correlation between lip movement and audio features
    sync_quality = np.corrcoef(lip_distances, audio_distance)[0, 1]
    
    return sync_quality

def extract_audio_features_from_video(video_file, sr=22050):
    """
    Extract audio features (pitch) from the audio of the video file.
    """
    # Load the audio from the video using moviepy
    video_clip = VideoFileClip(video_file)
    audio = video_clip.audio
    audio_samples = audio.to_soundarray(fps=sr)
    
    # Convert stereo to mono if needed
    if audio_samples.ndim == 2:
        audio_samples = audio_samples.mean(axis=1)
    
    # Extract pitch contour using librosa
    pitches, magnitudes = librosa.core.piptrack(y=audio_samples, sr=sr)
    pitch = np.max(pitches, axis=0)  # Get the max pitch for each frame
    
    return pitch

def evaluate_visual_speech_quality(video_path):
    """
    Evaluates the visual speech quality by comparing lip movements with audio features from the video.
    """
    # Extract audio features from the video
    audio_features = extract_audio_features_from_video(video_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    
    lip_movements = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract lip features from each frame
        lip_points = extract_lip_features(frame)
        
        if len(lip_points) > 0:
            lip_movements.append(lip_points)

        frame_count += 1

    # Close video capture
    cap.release()

    # We need to ensure the audio features and lip movements are of similar length
    # Sync video frames with audio frames based on FPS
    video_duration = frame_count / fps
    audio_duration = len(audio_features) / 22050  # Assuming default sample rate of 22050 Hz
    min_length = min(len(audio_features), len(lip_movements))

    # Resize or trim the features to match length (you may need to sync by time)
    audio_features = audio_features[:min_length]
    lip_movements = np.array(lip_movements[:min_length])

    # Evaluate lip sync quality
    sync_quality = compute_lip_sync_quality(lip_movements, audio_features, fps)
    return sync_quality

# Example usage:
video_path = 'agni.mp4'  # Path to the input video
sync_quality = evaluate_visual_speech_quality(video_path)

print(f'Visual Speech Quality: {sync_quality:.3f}')
