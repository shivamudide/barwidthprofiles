import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
from scipy import ndimage
from scipy.ndimage import rotate


def read_image_grayscale(path: str) -> np.ndarray:
    """Read image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return img


def compute_2d_fft(img: np.ndarray) -> tuple:
    """Compute 2D FFT of an image and return both complex and magnitude spectrum."""
    # Perform 2D FFT
    fft = np.fft.fft2(img)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Use log scale for better visualization (add 1 to avoid log(0))
    log_magnitude = np.log(magnitude_spectrum + 1)
    
    return fft_shifted, log_magnitude


def detect_tilt_angle(fft_magnitude: np.ndarray, mask_radius: int = 20) -> float:
    """
    Detect the tilt angle from the 2D FFT magnitude spectrum.
    Returns angle in degrees that the image should be rotated to correct tilt.
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create a mask to exclude the central DC component
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 > mask_radius**2
    
    # Apply mask
    masked_fft = fft_magnitude * mask
    
    # Find the dominant frequency peak (excluding center)
    max_idx = np.unravel_index(np.argmax(masked_fft), masked_fft.shape)
    peak_y, peak_x = max_idx
    
    # Calculate angle from center to peak
    dy = peak_y - center_y
    dx = peak_x - center_x
    
    # Angle in degrees (adjust by 90 degrees since we want bars to be vertical)
    angle = np.degrees(np.arctan2(dy, dx)) + 90
    
    return angle


def correct_image_tilt(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image to correct for tilt."""
    # Rotate image by the negative angle to correct tilt
    corrected = rotate(img, -angle, reshape=False, mode='constant', cval=0)
    return corrected


def radial_average(fft_magnitude: np.ndarray) -> tuple:
    """
    Compute radial average of 2D FFT magnitude spectrum.
    Returns radial distances and corresponding averaged magnitudes.
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create coordinate arrays
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center for each pixel
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Define radial bins
    max_radius = int(np.min([center_x, center_y]))
    radial_bins = np.arange(0, max_radius)
    
    # Compute radial average
    radial_profile = []
    for i in range(len(radial_bins) - 1):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
        if np.any(mask):
            radial_profile.append(np.mean(fft_magnitude[mask]))
        else:
            radial_profile.append(0)
    
    return radial_bins[:-1], np.array(radial_profile)


def angular_average(fft_magnitude: np.ndarray, num_angles: int = 180) -> tuple:
    """
    Compute angular average of 2D FFT magnitude spectrum.
    Returns angles and corresponding averaged magnitudes.
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create coordinate arrays
    y, x = np.ogrid[:h, :w]
    
    # Calculate angle from center for each pixel
    angles = np.degrees(np.arctan2(y - center_y, x - center_x))
    angles = (angles + 180) % 360  # Convert to 0-360 range
    
    # Define angular bins
    angle_bins = np.linspace(0, 360, num_angles + 1)
    
    # Compute angular average
    angular_profile = []
    for i in range(len(angle_bins) - 1):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        if np.any(mask):
            angular_profile.append(np.mean(fft_magnitude[mask]))
        else:
            angular_profile.append(0)
    
    return angle_bins[:-1], np.array(angular_profile)


def plot_comprehensive_analysis(img: np.ndarray, img_corrected: np.ndarray, 
                               fft_original: np.ndarray, fft_corrected: np.ndarray,
                               radial_dist: np.ndarray, radial_prof: np.ndarray,
                               angular_bins: np.ndarray, angular_prof: np.ndarray,
                               title: str, tilt_angle: float):
    """Plot comprehensive FFT analysis including tilt correction and 1D profiles."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image: {title}')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # Tilt-corrected image
    ax2 = plt.subplot(3, 4, 2)
    plt.imshow(img_corrected, cmap='gray')
    plt.title(f'Tilt Corrected: {title}\n(Rotated by {tilt_angle:.1f}°)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # Original 2D FFT
    ax3 = plt.subplot(3, 4, 5)
    plt.imshow(fft_original, cmap='hot')
    plt.title('Original 2D FFT')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')
    
    # Corrected 2D FFT
    ax4 = plt.subplot(3, 4, 6)
    plt.imshow(fft_corrected, cmap='hot')
    plt.title('Tilt-Corrected 2D FFT')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')
    
    # Radial average
    ax5 = plt.subplot(3, 4, 9)
    plt.plot(radial_dist, radial_prof, 'b-', linewidth=2)
    plt.title('Radial Average (1D)')
    plt.xlabel('Radial Frequency (pixels⁻¹)')
    plt.ylabel('Average Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Angular average
    ax6 = plt.subplot(3, 4, 10)
    plt.plot(angular_bins, angular_prof, 'r-', linewidth=2)
    plt.title('Angular Average (1D)')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Average Magnitude')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    
    # Radial profile (log scale)
    ax7 = plt.subplot(3, 4, 11)
    plt.semilogy(radial_dist, radial_prof, 'b-', linewidth=2)
    plt.title('Radial Average (Log Scale)')
    plt.xlabel('Radial Frequency (pixels⁻¹)')
    plt.ylabel('Average Magnitude (log)')
    plt.grid(True, alpha=0.3)
    
    # Angular profile (polar plot)
    ax8 = plt.subplot(3, 4, 12, projection='polar')
    theta = np.radians(angular_bins)
    plt.plot(theta, angular_prof, 'r-', linewidth=2)
    plt.title('Angular Distribution\n(Polar Plot)')
    
    plt.tight_layout()
    return fig


def analyze_all_unzoomed_images_advanced():
    """Advanced analysis of all images with tilt correction and 1D projections."""
    # Define paths
    unzoomed_dir = Path("images/unzoomed")
    output_dir = Path("outputs/fft_analysis_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg files in unzoomed directory
    image_files = list(unzoomed_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No .jpg files found in {unzoomed_dir}")
        return
    
    print(f"Found {len(image_files)} images to analyze with advanced FFT processing:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # Storage for summary data
    all_results = []
    
    # Process each image
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        
        # Read image
        img = read_image_grayscale(str(img_path))
        print(f"  Image dimensions: {img.shape[0]} x {img.shape[1]} pixels")
        
        # Compute original FFT
        fft_complex_orig, fft_mag_orig = compute_2d_fft(img)
        
        # Detect tilt angle
        tilt_angle = detect_tilt_angle(fft_mag_orig)
        print(f"  Detected tilt angle: {tilt_angle:.2f} degrees")
        
        # Correct tilt
        img_corrected = correct_image_tilt(img, tilt_angle)
        
        # Compute corrected FFT
        fft_complex_corr, fft_mag_corr = compute_2d_fft(img_corrected)
        
        # Compute 1D profiles from corrected FFT
        radial_dist, radial_prof = radial_average(fft_mag_corr)
        angular_bins, angular_prof = angular_average(fft_mag_corr)
        
        print(f"  Computed radial profile: {len(radial_dist)} points")
        print(f"  Computed angular profile: {len(angular_bins)} points")
        
        # Store results
        result = {
            'name': img_path.stem,
            'tilt_angle': tilt_angle,
            'radial_dist': radial_dist,
            'radial_prof': radial_prof,
            'angular_bins': angular_bins,
            'angular_prof': angular_prof,
            'img_shape': img.shape
        }
        all_results.append(result)
        
        # Create comprehensive plot
        fig = plot_comprehensive_analysis(
            img, img_corrected, fft_mag_orig, fft_mag_corr,
            radial_dist, radial_prof, angular_bins, angular_prof,
            img_path.stem, tilt_angle
        )
        
        # Save individual comprehensive plot
        output_path = output_dir / f"{img_path.stem}_advanced_fft_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")
    
    # Create comparison plot of 1D profiles
    print(f"\nCreating 1D profile comparison plots...")
    
    # Radial profiles comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for result in all_results:
        ax1.plot(result['radial_dist'], result['radial_prof'], 
                label=result['name'], linewidth=2)
        ax2.semilogy(result['radial_dist'], result['radial_prof'], 
                    label=result['name'], linewidth=2)
    
    ax1.set_title('Radial Profiles Comparison (Linear)')
    ax1.set_xlabel('Radial Frequency (pixels⁻¹)')
    ax1.set_ylabel('Average Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Radial Profiles Comparison (Log Scale)')
    ax2.set_xlabel('Radial Frequency (pixels⁻¹)')
    ax2.set_ylabel('Average Magnitude (log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    radial_comparison_path = output_dir / "radial_profiles_comparison.png"
    fig.savefig(radial_comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved radial comparison: {radial_comparison_path}")
    
    # Angular profiles comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for result in all_results:
        ax1.plot(result['angular_bins'], result['angular_prof'], 
                label=result['name'], linewidth=2)
    
    ax1.set_title('Angular Profiles Comparison')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Average Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    
    # Polar plot
    ax2 = plt.subplot(122, projection='polar')
    for result in all_results:
        theta = np.radians(result['angular_bins'])
        ax2.plot(theta, result['angular_prof'], label=result['name'], linewidth=2)
    ax2.set_title('Angular Profiles (Polar)')
    ax2.legend()
    
    plt.tight_layout()
    angular_comparison_path = output_dir / "angular_profiles_comparison.png"
    fig.savefig(angular_comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved angular comparison: {angular_comparison_path}")
    
    # Print summary
    print(f"\n=== Advanced FFT Analysis Summary ===")
    for result in all_results:
        print(f"{result['name']}:")
        print(f"  Tilt angle corrected: {result['tilt_angle']:.2f}°")
        print(f"  Image shape: {result['img_shape']}")
        print(f"  Radial profile peak: {np.max(result['radial_prof']):.2f}")
        print(f"  Angular profile peak: {np.max(result['angular_prof']):.2f}")


def plot_image_and_fft(img: np.ndarray, fft_magnitude: np.ndarray, title: str):
    """Plot original image and its 2D FFT side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Original Image: {title}')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # FFT magnitude spectrum
    ax2.imshow(fft_magnitude, cmap='hot')
    ax2.set_title(f'2D FFT Magnitude Spectrum: {title}')
    ax2.set_xlabel('Frequency X')
    ax2.set_ylabel('Frequency Y')
    
    plt.tight_layout()
    return fig


def analyze_all_unzoomed_images():
    """Analyze all images in the unzoomed folder."""
    # Define paths
    unzoomed_dir = Path("images/unzoomed")
    output_dir = Path("outputs/fft_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg files in unzoomed directory
    image_files = list(unzoomed_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No .jpg files found in {unzoomed_dir}")
        return
    
    print(f"Found {len(image_files)} images to analyze:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # Process each image
    all_ffts = []
    image_names = []
    
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        
        # Read image
        img = read_image_grayscale(str(img_path))
        print(f"  Image dimensions: {img.shape[0]} x {img.shape[1]} pixels")
        
        # Compute FFT
        _, fft_magnitude = compute_2d_fft(img)
        all_ffts.append(fft_magnitude)
        image_names.append(img_path.stem)
        
        # Create individual plot
        fig = plot_image_and_fft(img, fft_magnitude, img_path.stem)
        
        # Save individual plot
        output_path = output_dir / f"{img_path.stem}_fft_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")
    
    # Create comparison plot with all FFTs
    print(f"\nCreating comparison plot with all {len(all_ffts)} FFTs...")
    
    fig, axes = plt.subplots(2, len(all_ffts), figsize=(5*len(all_ffts), 10))
    if len(all_ffts) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (img_path, fft_mag, name) in enumerate(zip(image_files, all_ffts, image_names)):
        # Original image
        img = read_image_grayscale(str(img_path))
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original: {name}')
        axes[0, i].set_xlabel('X (pixels)')
        axes[0, i].set_ylabel('Y (pixels)')
        
        # FFT
        axes[1, i].imshow(fft_mag, cmap='hot')
        axes[1, i].set_title(f'2D FFT: {name}')
        axes[1, i].set_xlabel('Frequency X')
        axes[1, i].set_ylabel('Frequency Y')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = output_dir / "all_images_fft_comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot: {comparison_path}")
    
    # Print summary statistics
    print(f"\n=== FFT Analysis Summary ===")
    for i, (name, fft_mag) in enumerate(zip(image_names, all_ffts)):
        max_freq = np.max(fft_mag)
        mean_freq = np.mean(fft_mag)
        print(f"{name}:")
        print(f"  Max FFT magnitude: {max_freq:.2f}")
        print(f"  Mean FFT magnitude: {mean_freq:.2f}")
        print(f"  FFT shape: {fft_mag.shape}")


if __name__ == "__main__":
    print("Starting advanced 2D FFT analysis with tilt correction and 1D projections...")
    analyze_all_unzoomed_images_advanced()
    print("\nAdvanced FFT analysis complete!") 