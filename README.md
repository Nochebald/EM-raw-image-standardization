# CLAHE and Denoising Pre-processing Pipeline

This repository contains the Python-based image pre-processing script (`pre-processing.py`) utilized to standardize heterogeneous electron microscopy (EM) datasets (Palade, Uranyl-free, and Ellisman preparations) for robust deep learning segmentation. 

## Dependencies and Fallback Mechanism
This script is designed to execute seamlessly within the native ZEISS Arivis Pro Python environment.

* **Optimal Performance:** For the highest quality CLAHE (Contrast Limited Adaptive Histogram Equalization) and denoising, the script natively attempts to utilize **OpenCV** (`cv2`).
* **Automatic Fallback:** Because internal Arivis Python configurations can vary between laboratories, the script includes a built-in safety fallback. If OpenCV is not detected in your Arivis environment, the script will automatically switch to a native NumPy-based Gaussian blur and equalization method. This ensures the Operations pipeline will successfully execute without crashing, regardless of your local setup.

## Usage within ZEISS Arivis Pro
This script utilizes the `arivis_operation` API and is designed to run directly within the software's image processing pipeline.

1. Open your dataset in Arivis Pro.
2. Open the **Analysis Panel**.
3. Configure the **Input ROI** to your needs
4. Press the **Add operation** button to get a list of available operations.
5. Add the **Python Image Filter** operator to your pipeline.
6. Load the `pre-processing.py` script into this operator.
7. Execute the pipeline to output the standardized image layer.

## Advanced Configuration (Customizable Parameters)
While the script is optimized for peripheral nerve ultrastructure out-of-the-box, advanced users can fine-tune several parameters directly within the `pre-processing.py` script to suit different tissue types, staining protocols, or resolutions.

### 1. Memory and Performance Settings
Located in the `===== CONFIGURATION =====` block, these dictate how the script calculates the dataset's global mean and standard deviation:
* **`USE_EXACT_STATISTICS` (Default: `False`):** Set to `True` to load the entire volume into RAM for mathematically perfect global statistics. Keep `False` to use memory-efficient random sampling (highly recommended for large EM volumes).
* **`MAX_SLICES_FOR_EXACT` (Default: `100`):** The slice-count threshold. If your dataset has fewer slices than this, the script automatically defaults to exact statistics.
* **`max_samples` (Default: `500000`):** The total number of pixels to randomly sample across the volume when estimating global statistics. Increase for higher accuracy on massive datasets, decrease to save RAM.

### 2. Denoising Strength (OpenCV mode)
Located in `def denoise_image(image):`
* **`h=10`**: The filter strength. Higher values remove more noise but blur image details. 
* **`templateWindowSize=7` & `searchWindowSize=21`**: Compute window sizes. Increase for better denoising at the cost of significantly longer processing times.

### 3. Local Contrast (CLAHE)
Located in `def apply_clahe(image):`
* **`clipLimit=3.0`**: Sets the contrast limit for histogram equalization. If your resulting images look artificially noisy, lower this value (e.g., `2.0`). If they are too flat, raise it.
* **`tileGridSize=(8, 8)`**: The size of the grid for local contrast calculation. Larger grids factor in larger tissue regions for contrast balancing.

### 4. Normalization and Sharpening
* **Z-Score Clipping:** The script scales the data assuming most biological signal falls within ±3 standard deviations. If you have extreme outliers, you can widen this range.
* **Sharpening Matrix:** A 3x3 kernel is applied to make organelle membranes pop. To increase sharpness, increase the center value (currently `5.0`) and make the surrounding negative values correspondingly lower to ensure the matrix sum remains `1.0`.

## Running the Pipeline Outside of Arivis Pro
The provided `pre-processing.py` relies on the proprietary `arivis_operation` API to read and write volume data. However, the core algorithmic steps (Denoise -> Normalize -> CLAHE -> Invert -> Sharpen) are built on standard Python libraries (`numpy` and `cv2`) and can easily be adapted for standalone use.

To use this pipeline in a standard Jupyter Notebook, terminal script, or open-source pipeline (like Napari or ImageJ/Fiji via Jython), you can replace the Arivis-specific I/O with standard libraries like `tifffile`:

```python
import numpy as np
import cv2
import tifffile # pip install tifffile

# [PASTE denoise_image() and apply_clahe() FUNCTIONS FROM SCRIPT HERE]

def process_standalone_volume(input_path, output_path):
    print("Loading volume...")
    volume = tifffile.imread(input_path)
    
    print("Calculating global statistics...")
    global_mean = np.mean(volume.astype(np.float64))
    global_std = np.std(volume.astype(np.float64))
    
    processed_volume = np.zeros_like(volume, dtype=np.uint8)
    
    print("Processing slices...")
    for z in range(volume.shape[0]):
        slice_2d = volume[z]
        
        # 1. Denoise
        denoised = denoise_image(slice_2d)
        
        # 2. Global Z-Score Normalization
        normalized = (denoised.astype(np.float64) - global_mean) / (global_std + 1e-5)
        normalized = np.clip((normalized + 3) / 6 * 255, 0, 255).astype(np.uint8)
        
        # 3. CLAHE
        enhanced = apply_clahe(normalized)
        
        # 4. Invert
        inverted = 255 - enhanced
        
        # 5. Sharpen
        kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(inverted, -1, kernel_sharpen)
        processed_volume[z] = np.clip(sharpened, 0, 255).astype(np.uint8)
        
    print("Saving processed volume...")
    tifffile.imwrite(output_path, processed_volume)
    print("Done!")

# Example usage:
# process_standalone_volume('raw_em_dataset.tif', 'standardized_dataset.tif')
