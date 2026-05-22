Dependencies and Fallback Mechanism

This script is designed to execute seamlessly within the native ZEISS Arivis Pro Python environment.

    Optimal Performance: For the highest quality CLAHE (Contrast Limited Adaptive Histogram Equalization) and denoising, the script natively attempts to utilize OpenCV (cv2).

    Automatic Fallback: Because internal Arivis Python configurations can vary between laboratories, the script includes a built-in safety fallback. If OpenCV is not detected in your Arivis environment, the script will automatically switch to a native NumPy-based Gaussian blur and equalization method. This ensures the Operations pipeline will successfully execute without crashing, regardless of your local setup.

Usage within ZEISS Arivis Pro

This script utilizes the arivis_operation API and is designed to run directly within the software's image processing pipeline.

    Open your dataset in Arivis Pro.

    Open the Operations panel to configure your analysis pipeline.

    Add the Python Image Filter operator to your pipeline.

    Load the pre-processing.py script into this operator.

    Execute the pipeline to output the standardized image layer.
