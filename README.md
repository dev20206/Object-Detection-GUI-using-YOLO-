# Object Detection GUI using YOLO

This project implements a Graphical User Interface (GUI) for object detection using the YOLO (You Only Look Once) deep learning model. It allows users to open an image file and performs real-time object detection, displaying bounding boxes and labels for detected objects. Additionally, it provides visualizations for object detection confidence, object count distribution, object size vs confidence, and class distribution.

## Features

- **Image Selection:** Users can open an image file from their local system using the "Open Image" button.
- **Real-time Object Detection:** Utilizes the YOLO model to perform real-time object detection on the selected image.
- **Bounding Boxes and Labels:** Draws bounding boxes around detected objects and displays corresponding labels.
- **Visualizations:**
  - Object Detection Confidence: Displays the confidence score for each detected object.
  - Object Count Distribution: Shows the frequency distribution of the number of objects per class.
  - Object Size vs Confidence: Scatter plot depicting the relationship between object size (area) and confidence score.
  - Class Distribution: Bar chart illustrating the distribution of object classes detected in the image.
- **Save Image:** Provides an option to save the processed image with bounding boxes and labels.

## Technical Tools Used

- **Programming Languages:** Python
- **Libraries/Frameworks:** OpenCV, NumPy, Matplotlib, tkinter (for GUI), PIL (Python Imaging Library)
- **Deep Learning Model:** YOLO (You Only Look Once)
- **Development Tools:** PyCharm IDE, Git

## Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `main.py` script using Python.
4. Click on the "Open Image" button to select an image for object detection.
5. View the detected objects with bounding boxes and labels, along with visualizations.
6. Optionally, save the processed image using the "Save Image" button.

