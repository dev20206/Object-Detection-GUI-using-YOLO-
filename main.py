import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Load YOLO model
yolo = cv2.dnn.readNet("C:\\Users\\devv2\\Desktop\\project\\yolov3.weights",
                       "C:\\Users\\devv2\\Desktop\\project\\yolov3.cfg")

# Load classes from coco.names
with open("C:\\Users\\devv2\\Desktop\\project\\coco.names", 'r') as f:
    classes = f.read().splitlines()

# Initialize class_ids and confidences as global variables
class_ids = []
confidences = []


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        process_image(img)


def process_image(img):
    global class_ids, confidences
    class_ids = []
    confidences = []

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

    height, width, _ = img.shape
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

    # Display the processed image with bounding boxes and labels
    display_image(img, boxes, indices)


def display_image(img, boxes, indices):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    display_img = img.copy()

    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i], 2))
        color = colors[i]

        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 5)
        cv2.putText(display_img, label + " " + confi, (x, y + 30), font, font_scale, (255, 255, 255), font_thickness)

    # Resize the image while maintaining the aspect ratio
    max_height = 600
    img_height, img_width, _ = display_img.shape
    aspect_ratio = img_width / img_height

    new_height = min(img_height, max_height)
    new_width = int(aspect_ratio * new_height)

    display_img = cv2.resize(display_img, (new_width, new_height))

    # Plotting Object Detection Confidence
    labels = [str(classes[class_ids[i]]) for i in indices]
    confidence_values = [round(confidences[i], 2) for i in indices]

    plt.figure(figsize=(8, 4))
    plt.barh(labels, confidence_values, color='blue')
    plt.xlabel('Confidence Score')
    plt.title('Object Detection Confidence')
    plt.show()

    # Plotting Object Count Distribution
    class_counts = {}
    for class_id in class_ids:
        class_name = classes[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    plt.figure(figsize=(8, 4))
    plt.hist(list(class_counts.values()), bins=range(1, max(class_counts.values()) + 1), color='orange',
             edgecolor='black')
    plt.xlabel('Number of Objects per Class')
    plt.ylabel('Frequency')
    plt.title('Object Count Distribution')
    plt.show()

    # Plotting Object Size vs Confidence
    object_sizes = [box[2] * box[3] for box in boxes]

    plt.figure(figsize=(8, 4))
    plt.scatter(object_sizes, confidences, color='blue', alpha=0.5)
    plt.xlabel('Object Size (Area)')
    plt.ylabel('Confidence Score')
    plt.title('Object Size vs Confidence')
    plt.show()

    # Plotting Class Distribution
    class_counts = {}
    for class_id in class_ids:
        class_name = classes[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    plt.figure(figsize=(8, 4))
    plt.bar(class_counts.keys(), class_counts.values(), color='green')
    plt.xlabel('Object Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Create a new window for image display
    window = tk.Toplevel(root)
    window.title("Object Detection Result")

    # Display the image in the new window
    image_label = tk.Label(window)
    image_label.pack()

    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    display_img = Image.fromarray(display_img)
    display_img = ImageTk.PhotoImage(display_img)
    image_label.img = display_img
    image_label.config(image=display_img)

    # Add "Save Image" button
    save_button = ttk.Button(window, text="Save Image", command=lambda: save_image(display_img))
    save_button.pack()


def save_image(img):
    pil_image = ImageTk.getimage(img)
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        pil_image.save(file_path)


# Create the main Tkinter GUI window
root = tk.Tk()
root.title("Object Detection GUI")

root.iconbitmap('C:\\Users\\devv2\\Desktop\\project\\big-data.asp_final-d2382542261e4dafa0a1faa05ea2fcce.png')
root.minsize(530, 310)
root.configure(background='black')

# Image on the left
img = Image.open('C:\\Users\\devv2\\Downloads\\Bright happy birthday instagram gif feed post  - Copy.png')
resized_img = img.resize((300, 300))
img = ImageTk.PhotoImage(resized_img)
img_label = tk.Label(root, image=img)
img_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

# Style for ttk buttons
style = ttk.Style()
style.configure('TButton', font=('Arial Narrow', 12), padding=5, background='white', foreground='black')

# Button to open an image on the right
open_button = ttk.Button(root, text="Open Image", command=open_image)
open_button.grid(row=0, column=1, pady=10)

# Label to provide instructions on the right
instruction_label = tk.Label(root,
                             text="1. Click 'Open Image' to select an image.\n2. View the object detection result.")
instruction_label.grid(row=1, column=1, pady=10)

# Run the Tkinter event loop
root.mainloop()
