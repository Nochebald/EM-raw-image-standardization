import arivis_operation
import numpy as np

# Try to import OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ==============================================================================
# ========================== GLOBAL CONFIGURATION ==============================
# ==============================================================================

# --- Memory Management ---
# Set to True to load all slices (exact statistics, high memory)
# Set to False to use sampling (approximate statistics, low memory)
USE_EXACT_STATISTICS = False  # Change to True for small stacks
MAX_SLICES_FOR_EXACT = 100    # Auto-switch to exact mode if stack is small

# --- Filter & Enhancement Parameters ---
# Grid size for local contrast (CLAHE). Increase for larger structural variations.
CLAHE_GRID_SIZE = (8, 8)

# Sharpening matrix (Kernel). Applied to enhance organelle membranes.
# Note: The matrix should sum to 1.0 to preserve overall image brightness.
# Default is a 3x3 kernel with a center strength of 5.0.
SHARPEN_KERNEL = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5,  5.0, -0.5],
    [-0.5, -0.5, -0.5]
], dtype=np.float32)

# ==============================================================================


def denoise_image(image):
    """
    Remove noise before processing
    """
    if HAS_OPENCV:
        # OpenCV's Non-local Means Denoising (very effective)
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised
    else:
        # Simple Gaussian blur fallback
        h, w = image.shape
        scale = 2
        small = image[::scale, ::scale].astype(np.float32)
        
        # 3x3 averaging
        kernel = np.ones((3, 3)) / 9.0
        pad = 1
        padded = np.pad(small, pad, mode='reflect')
        
        smoothed = np.zeros_like(small)
        for i in range(small.shape[0]):
            for j in range(small.shape[1]):
                smoothed[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        # Upsample
        result = np.repeat(np.repeat(smoothed, scale, axis=0), scale, axis=1)
        result = result[:h, :w]
        return result.astype(np.uint8)


def apply_clahe(image):
    """
    Apply CLAHE for local contrast enhancement
    """
    if HAS_OPENCV:
        # OpenCV CLAHE - optimal settings for EM images, utilizing global grid size
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=CLAHE_GRID_SIZE)
        enhanced = clahe.apply(image)
        return enhanced
    else:
        # Manual histogram equalization fallback
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        cdf_normalized = (cdf - cdf_min) * 255 / (cdf.max() - cdf_min + 1e-10)
        equalized = cdf_normalized[image].astype(np.uint8)
        return equalized


# Get the operation context
context = arivis_operation.Operation.get_context()
input_data = context.get_input()
output_data = context.get_output()
bounds = input_data.get_bounds()

channel = 0
timepoint = bounds.t1
x_start, x_end = bounds.x1, bounds.x2
y_start, y_end = bounds.y1, bounds.y2
z_start, z_end = bounds.z1, bounds.z2
width = x_end - x_start
height = y_end - y_start


# FIRST PASS: Calculate global statistics
z = z_start
slice_count = 0

# Count slices first
temp_z = z_start
while True:
    try:
        start = (x_start, y_start, temp_z)
        size = (width, height, 1)
        input_data.read_imagedata((start, size), channel, timepoint)
        slice_count += 1
        temp_z += 1
        if temp_z > z_end + 10:
            break
    except:
        break

# Decide whether to use exact or sampling method
use_exact = USE_EXACT_STATISTICS or (slice_count <= MAX_SLICES_FOR_EXACT)

if use_exact:
    # EXACT METHOD: Load all slices
    all_slices = []
    z = z_start
    
    while True:
        start = (x_start, y_start, z)
        size = (width, height, 1)
        
        try:
            input_slice = input_data.read_imagedata((start, size), channel, timepoint)
        except:
            break
        
        if input_slice.ndim == 3:
            slice_2d = input_slice[0, :, :]
        else:
            slice_2d = input_slice
        
        if slice_2d.size == 0:
            break
        
        all_slices.append(slice_2d)
        
        z += 1
        if z > z_end + 10:
            break
    
    stack = np.array(all_slices, dtype=np.float64)
    global_mean = np.mean(stack)
    global_std = np.std(stack)
    del stack
    del all_slices

else:
    # SAMPLING METHOD: Sample pixels for large stacks
    sample_values = []
    max_samples = 500000
    samples_per_slice = max_samples // slice_count
    
    z = z_start
    while True:
        start = (x_start, y_start, z)
        size = (width, height, 1)
        
        try:
            input_slice = input_data.read_imagedata((start, size), channel, timepoint)
        except:
            break
        
        if input_slice.ndim == 3:
            slice_2d = input_slice[0, :, :]
        else:
            slice_2d = input_slice
        
        if slice_2d.size == 0:
            break
        
        flat = slice_2d.flatten().astype(np.float64)
        if len(flat) <= samples_per_slice:
            sample_values.extend(flat)
        else:
            indices = np.random.choice(len(flat), samples_per_slice, replace=False)
            sample_values.extend(flat[indices])
        
        z += 1
        if z > z_end + 10:
            break
    
    sample_array = np.array(sample_values, dtype=np.float64)
    global_mean = np.mean(sample_array)
    global_std = np.std(sample_array)
    del sample_values
    del sample_array

# SECOND PASS: Process each slice with normalization (streaming approach)
z = z_start

while True:
    start = (x_start, y_start, z)
    size = (width, height, 1)
    
    # Read slice
    try:
        input_slice = input_data.read_imagedata((start, size), channel, timepoint)
    except:
        break
    
    if input_slice.ndim == 3:
        slice_2d = input_slice[0, :, :]
    else:
        slice_2d = input_slice
    
    if slice_2d.size == 0:
        break
    
    # ===== STEP 1: DENOISE =====
    denoised = denoise_image(slice_2d)
    
    # ===== STEP 2: Z-SCORE NORMALIZATION =====
    # Makes all slices have the same mean and std
    normalized = (denoised.astype(np.float64) - global_mean) / (global_std + 1e-5)
    
    # Scale to 0-255 range (center at 128, ±3 std covers most data)
    normalized = np.clip((normalized + 3) / 6 * 255, 0, 255).astype(np.uint8)
    
    # ===== STEP 3: CLAHE FOR LOCAL CONTRAST =====
    enhanced = apply_clahe(normalized)
    
    # ===== STEP 4: INVERT (dark becomes bright) =====
    inverted = 255 - enhanced
    
    # ===== STEP 5: FINAL SHARPENING (optional but helpful) =====
    if HAS_OPENCV:
        # Mild sharpening utilizing the global SHARPEN_KERNEL
        sharpened = cv2.filter2D(inverted, -1, SHARPEN_KERNEL)
        final_output = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        final_output = inverted
    
    # Prepare output
    if input_slice.ndim == 3:
        output_slice = final_output[np.newaxis, :, :]
    else:
        output_slice = final_output
    
    # Write back
    try:
        output_data.write_imagedata(output_slice, (start, size), channel, timepoint)
    except:
        break
    
    z += 1
