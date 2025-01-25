import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def evaluate_enhancement(original_video_path, enhanced_video_path):
    """
    Evaluates the quality of video enhancement by comparing original and enhanced videos
    using metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
    
    Args:
        original_video_path: Path to the original low-resolution video
        enhanced_video_path: Path to the enhanced high-resolution video
        
    Returns:
        Dictionary containing average PSNR and SSIM scores
    """
    
    # Check if files exist
    if not os.path.exists(original_video_path):
        raise FileNotFoundError(f"Original video not found: {original_video_path}")
    if not os.path.exists(enhanced_video_path):
        raise FileNotFoundError(f"Enhanced video not found: {enhanced_video_path}")
    
    # Open both videos
    cap_original = cv2.VideoCapture(original_video_path)
    cap_enhanced = cv2.VideoCapture(enhanced_video_path)
    
    # Check if videos opened successfully
    if not cap_original.isOpened():
        raise ValueError(f"Could not open original video: {original_video_path}")
    if not cap_enhanced.isOpened():
        raise ValueError(f"Could not open enhanced video: {enhanced_video_path}")
    
    # Print video properties for debugging
    print(f"Original video frames: {int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Enhanced video frames: {int(cap_enhanced.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    original_psnr = []
    original_ssim = []
    enhanced_psnr = []
    enhanced_ssim = []
    frame_count = 0
    frame_numbers = []
    
    # Reference frame (first frame of original video)
    _, reference_frame = cap_original.read()
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    # Reset video capture
    cap_original.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        # Read frames from both videos
        ret1, frame_original = cap_original.read()
        ret2, frame_enhanced = cap_enhanced.read()
        
        if not ret1 or not ret2:
            break
            
        # Convert frames to grayscale
        gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size
        if gray_original.shape != gray_enhanced.shape:
            gray_enhanced = cv2.resize(gray_enhanced, (gray_original.shape[1], gray_original.shape[0]))
        
        # Calculate metrics against reference frame
        original_psnr.append(psnr(reference_gray, gray_original))
        original_ssim.append(ssim(reference_gray, gray_original))
        enhanced_psnr.append(psnr(reference_gray, gray_enhanced))
        enhanced_ssim.append(ssim(reference_gray, gray_enhanced))
        
        frame_numbers.append(frame_count)
        frame_count += 1
    
    # Release video captures
    cap_original.release()
    cap_enhanced.release()
    
    # Calculate time points
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    time_points = [i/fps for i in frame_numbers]
    
    # Take fewer points for smoother interpolation
    step = 10
    time_subset = time_points[::step]
    original_psnr_subset = original_psnr[::step]
    enhanced_psnr_subset = enhanced_psnr[::step]
    original_ssim_subset = original_ssim[::step]
    enhanced_ssim_subset = enhanced_ssim[::step]

    # Generate smooth curves
    T_smooth = np.linspace(min(time_points), max(time_points), 500)
    
    # Create smooth splines
    spl_psnr_orig = make_interp_spline(time_subset, original_psnr_subset, k=3)
    spl_psnr_enh = make_interp_spline(time_subset, enhanced_psnr_subset, k=3)
    spl_ssim_orig = make_interp_spline(time_subset, original_ssim_subset, k=3)
    spl_ssim_enh = make_interp_spline(time_subset, enhanced_ssim_subset, k=3)
    
    # Generate smooth curves
    psnr_smooth_orig = spl_psnr_orig(T_smooth)
    psnr_smooth_enh = spl_psnr_enh(T_smooth)
    ssim_smooth_orig = spl_ssim_orig(T_smooth)
    ssim_smooth_enh = spl_ssim_enh(T_smooth)

    # Plotting
    plt.figure(figsize=(15, 6))
    
    # PSNR subplot
    plt.subplot(1, 2, 1)
    plt.plot(T_smooth, psnr_smooth_orig, 'b-', label='Original Video', linewidth=2)
    plt.plot(T_smooth, psnr_smooth_enh, 'r-', label='Enhanced Video', linewidth=2)
    plt.title('PSNR over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('PSNR (dB)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # SSIM subplot
    plt.subplot(1, 2, 2)
    plt.plot(T_smooth, ssim_smooth_orig, 'b-', label='Original Video', linewidth=2)
    plt.plot(T_smooth, ssim_smooth_enh, 'r-', label='Enhanced Video', linewidth=2)
    plt.title('SSIM over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('SSIM', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the quality metrics plot
    plot_path = os.path.join(plots_dir, f'quality_metrics_{os.path.basename(original_video_path).split(".")[0]}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'average_psnr_original': np.mean(original_psnr),
        'average_psnr_enhanced': np.mean(enhanced_psnr),
        'average_ssim_original': np.mean(original_ssim),
        'average_ssim_enhanced': np.mean(enhanced_ssim),
        'frames_evaluated': frame_count
    }
    
    return results

def evaluate_multiple_videos(original_dir, enhanced_dir):
    """
    Evaluates multiple pairs of original and enhanced videos
    
    Args:
        original_dir: Directory containing original videos
        enhanced_dir: Directory containing enhanced videos
    
    Returns:
        List of dictionaries containing results for each video pair
    """
    video_results = []
    video_names = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
    
    for video_name in video_names:
        original_path = os.path.join(original_dir, video_name)
        enhanced_path = os.path.join(enhanced_dir, video_name)
        
        try:
            result = evaluate_enhancement(original_path, enhanced_path)
            result['video_name'] = video_name
            video_results.append(result)
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
    
    # Create comparative plots
    plt.figure(figsize=(12, 5))
    
    # PSNR comparison
    plt.subplot(1, 2, 1)
    video_numbers = range(1, len(video_results) + 1)
    psnr_values = [r['average_psnr_original'] for r in video_results]
    plt.bar(video_numbers, psnr_values, color='blue', alpha=0.7)
    plt.title('Average PSNR Comparison')
    plt.xlabel('Video Number')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    
    # SSIM comparison
    plt.subplot(1, 2, 2)
    ssim_values = [r['average_ssim_original'] for r in video_results]
    plt.bar(video_numbers, ssim_values, color='red', alpha=0.7)
    plt.title('Average SSIM Comparison')
    plt.xlabel('Video Number')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparative metrics plot
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    comparative_plot_path = os.path.join(plots_dir, 'comparative_metrics.png')
    plt.savefig(comparative_plot_path)
    plt.close()
    
    # Create results table
    print("\nResults Table:")
    print("-" * 65)
    print(f"{'Video Name':<12} | {'PSNR (dB)':<15} | {'SSIM':<15} | {'Frames':<10}")
    print("-" * 65)
    for result in video_results:
        print(f"{result['video_name']:<12} | {result['average_psnr_original']:15.2f} | {result['average_ssim_original']:15.4f} | {result['frames_evaluated']:<10}")
    print("-" * 65)
    
    return video_results

# Remove the old example usage and replace with:
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_dir = os.path.join(current_dir, "original_videos")
    enhanced_dir = os.path.join(current_dir, "enhanced_videos")

    try:
        results = evaluate_multiple_videos(original_dir, enhanced_dir)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
