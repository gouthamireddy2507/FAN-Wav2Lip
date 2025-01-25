import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt

def compare_video_quality(original_video_path, enhanced_video_path):
    """
    Compares video quality before and after enhancement using PSNR and SSIM metrics.
    
    Args:
        original_video_path: Path to the original video
        enhanced_video_path: Path to the enhanced video
        
    Returns:
`        Dictionary containing PSNR and SSIM metrics
`    """
    
    # Check if files exist
    if not os.path.exists(original_video_path):
        raise FileNotFoundError(f"Original video not found: {original_video_path}")
    if not os.path.exists(enhanced_video_path):
        raise FileNotFoundError(f"Enhanced video not found: {enhanced_video_path}")
    
    # Open videos
    cap_original = cv2.VideoCapture(original_video_path)
    cap_enhanced = cv2.VideoCapture(enhanced_video_path)
    
    # Verify videos opened successfully
    if not cap_original.isOpened():
        raise ValueError(f"Could not open original video: {original_video_path}")
    if not cap_enhanced.isOpened():
        raise ValueError(f"Could not open enhanced video: {enhanced_video_path}")
    
    # Print video properties
    print(f"Original video frames: {int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Enhanced video frames: {int(cap_enhanced.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    psnr_before = []
    psnr_after = []
    ssim_scores = []
    frame_count = 0
    frame_numbers = []
    
    while True:
        ret1, frame_original = cap_original.read()
        ret2, frame_enhanced = cap_enhanced.read()
        
        if not ret1 or not ret2:
            break
            
        # Convert to grayscale
        gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size for comparison
        if gray_original.shape != gray_enhanced.shape:
            gray_enhanced = cv2.resize(gray_enhanced, (gray_original.shape[1], gray_original.shape[0]))
        
        # Calculate PSNR against a downscaled version
        downscaled = cv2.resize(gray_original, (gray_original.shape[1]//2, gray_original.shape[0]//2))
        upscaled = cv2.resize(downscaled, (gray_original.shape[1], gray_original.shape[0]))
        psnr_before_value = psnr(gray_original, upscaled)
        psnr_after_value = psnr(gray_original, gray_enhanced)
        ssim_value = ssim(gray_original, gray_enhanced)
        
        # Store values
        psnr_before.append(psnr_before_value)
        psnr_after.append(psnr_after_value)
        ssim_scores.append(ssim_value)
        frame_numbers.append(frame_count)
        frame_count += 1
    
    # Release videos
    cap_original.release()
    cap_enhanced.release()
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # PSNR comparison plot
    plt.subplot(1, 2, 1)
    plt.plot(frame_numbers, psnr_before, 'b-', label='Before Enhancement')
    plt.plot(frame_numbers, psnr_after, 'r-', label='After Enhancement')
    plt.title('PSNR Comparison')
    plt.xlabel('Frame Number')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()
    
    # SSIM plot
    plt.subplot(1, 2, 2)
    plt.plot(frame_numbers, ssim_scores, 'g-')
    plt.title('SSIM over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quality_comparison_metrics.png')
    plt.close()
    
    # Calculate averages
    results = {
        'average_psnr_before': np.mean(psnr_before),
        'average_psnr_after': np.mean(psnr_after),
        'average_ssim': np.mean(ssim_scores),
        'frames_evaluated': frame_count,
        'psnr_improvement': np.mean(psnr_after) - np.mean(psnr_before)
    }
    
    return results


def compare_multiple_videos(original_dir, enhanced_dir):
    """
    Compares quality metrics for multiple pairs of videos
    
    Args:
        original_dir: Directory containing original videos
        enhanced_dir: Directory containing enhanced videos
    
    Returns:
        List of dictionaries containing results for each video pair
    """
    video_results = []
    video_names = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']  # Modify as needed
    
    for video_name in video_names:
        original_path = os.path.join(original_dir, video_name)
        enhanced_path = os.path.join(enhanced_dir, video_name)
        
        try:
            result = compare_video_quality(original_path, enhanced_path)
            result['video_name'] = video_name
            video_results.append(result)
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
    
    # Create line plot for comparison
    plt.figure(figsize=(15, 6))
    
    # PSNR comparison as curves
    video_numbers = range(1, len(video_results) + 1)
    plt.plot(video_numbers, 
            [r['average_psnr_before'] for r in video_results], 
            'bo-', label='Before Enhancement', linewidth=2, markersize=8)
    plt.plot(video_numbers, 
            [r['average_psnr_after'] for r in video_results], 
            'ro-', label='After Enhancement', linewidth=2, markersize=8)
    
    plt.title('PSNR Comparison Across Videos')
    plt.xlabel('Video Number')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(video_numbers)  # Set x-axis ticks to video numbers
    
    plt.tight_layout()
    plt.savefig('psnr_comparison_across_videos.png')
    plt.close()
    
    # Print results table
    print("\nResults Table:")
    print("-" * 100)
    print(f"{'Video Name':<12} | {'PSNR Before':<15} | {'PSNR After':<15} | {'Improvement':<15} | {'SSIM':<15} | {'Frames':<10}")
    print("-" * 100)
    for result in video_results:
        print(f"{result['video_name']:<12} | "
              f"{result['average_psnr_before']:15.2f} | "
              f"{result['average_psnr_after']:15.2f} | "
              f"{result['psnr_improvement']:15.2f} | "
              f"{result['average_ssim']:15.4f} | "
              f"{result['frames_evaluated']:<10}")
    print("-" * 100)
    
    return video_results

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_dir = os.path.join(current_dir, "original_videos")
    enhanced_dir = os.path.join(current_dir, "enhanced_videos")

    try:
        results = compare_multiple_videos(original_dir, enhanced_dir)
        print("\nComparison completed successfully!")
        print("Generated plots: 'quality_comparison_metrics.png' and 'psnr_comparison_across_videos.png'")
    except Exception as e:
        print(f"Error occurred: {str(e)}") 