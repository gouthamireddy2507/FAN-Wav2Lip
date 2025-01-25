import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Paths to video folders
original_videos_path = "./original_videos"
enhanced_videos_path = "./enhanced_videos"

# Video file names
video_files = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]

# Function to read frames from a video
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# Store PSNR and SSIM values
psnr_values = {"original": [], "enhanced": []}
ssim_values = {"original": [], "enhanced": []}

for video_file in video_files:
    # Read frames from original and enhanced videos
    original_frames = read_video_frames(os.path.join(original_videos_path, video_file))
    enhanced_frames = read_video_frames(os.path.join(enhanced_videos_path, video_file))
    
    # Debugging: Print the number of frames read
    print(f"Original frames for {video_file}: {len(original_frames)}")
    print(f"Enhanced frames for {video_file}: {len(enhanced_frames)}")
    
    # Check if the videos have the same number of frames
    if len(original_frames) != len(enhanced_frames):
        print(f"Warning: Mismatch in frame count for {video_file}. Skipping this video.")
        continue  # Skip to the next video
    
    video_psnr = []
    video_ssim = []
    
    for orig_frame, enh_frame in zip(original_frames, enhanced_frames):
        # Compute PSNR and SSIM for each frame
        video_psnr.append(psnr(orig_frame, enh_frame, data_range=255))
        video_ssim.append(ssim(orig_frame, enh_frame, multichannel=True))
    
    # Store the mean values for each video
    psnr_values["original"].append(video_psnr)
    ssim_values["original"].append(video_ssim)
    psnr_values["enhanced"].append(video_psnr)
    ssim_values["enhanced"].append(video_ssim)

# Generate plots
def plot_metric_over_frames(metric_values, metric_name):
    for i, video_file in enumerate(video_files):
        frames = list(range(1, len(metric_values["original"][i]) + 1))
        smooth_frames = np.linspace(frames[0], frames[-1], 300)
        
        # Smooth original video metric curve
        original_spline = make_interp_spline(frames, metric_values["original"][i], k=3)
        original_smooth = original_spline(smooth_frames)
        
        # Smooth enhanced video metric curve
        enhanced_spline = make_interp_spline(frames, metric_values["enhanced"][i], k=3)
        enhanced_smooth = enhanced_spline(smooth_frames)
        
        plt.figure(figsize=(10, 6))
        plt.plot(smooth_frames, original_smooth, label="Original Video", color="blue")
        plt.plot(smooth_frames, enhanced_smooth, label="Enhanced Video", color="green")
        plt.title(f"{metric_name} Over Frames for Video {video_file}")
        plt.xlabel("Frame Number")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot PSNR and SSIM
plot_metric_over_frames(psnr_values, "PSNR")
plot_metric_over_frames(ssim_values, "SSIM")