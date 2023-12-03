import sys
import cv2
import os
import shutil
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox, QVBoxLayout, \
    QWidget, QSlider, QDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal

class ImageViewer(QDialog):
    def __init__(self, images):
        super().__init__()

        self.setWindowTitle("Detected Images")
        self.setFixedSize(800, 600)  # Set a fixed size for the window

        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.images = images
        self.current_index = 0

        self.show_image()

        self.setLayout(layout)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_current_image)
        layout.addWidget(self.save_button)

        self.setLayout(layout)


    def save_current_image(self):
        if 0 <= self.current_index < len(self.images):
            image_path = self.images[self.current_index]

            # Use QFileDialog to specify the save location
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                       "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)")

            if save_path:
                # Copy the current image to the specified save location
                shutil.copy(image_path, save_path)
                QMessageBox.information(self, "Image Saved", f"The image has been saved to {save_path}")

    def show_image(self):
        if 0 <= self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.images):
            self.current_index = 0
        self.show_image()

    def previous_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.images) - 1
        self.show_image()

class PersonDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Person Detection App")
        self.setGeometry(100, 100, 400, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        mp4_label = QLabel("Only MP4 files are allowed")
        layout.addWidget(mp4_label)

        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)

        self.browse_button = QPushButton("Upload MP4 File")
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        self.thumbnail_label = QLabel()  # QLabel to display the thumbnail
        layout.addWidget(self.thumbnail_label)

        self.extract_button = QPushButton("Extract Person Detected")
        self.extract_button.setStyleSheet("background-color: #007ACC; color: black; font-weight: bold;")
        self.extract_button.clicked.connect(self.extract_person)
        layout.addWidget(self.extract_button)

        layout.addStretch(1)  # Add some space to push buttons to the bottom

        self.central_widget.setLayout(layout)

        self.file_path = ""
        self.detected_images = []  # List to store paths of images with detected persons
        self.current_image_index = 0  # Index to track the current image being displayed

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MP4 File", "", "MP4 Files (*.mp4);;All Files (*)")

        if file_path:
            self.file_label.setText("File selected: " + file_path)
            self.file_path = file_path
            self.display_thumbnail()

    def display_thumbnail(self):
        if self.file_path.endswith(".mp4"):
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(self.file_path)

            # Check if the video file was successfully opened
            if not cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file")
                return

            # Read the first frame (thumbnail)
            ret, frame = cap.read()

            if ret:
                # Define a fixed size for the thumbnail
                thumbnail_size = (320, 240)  # You can adjust the size as needed

                # Resize the frame to the fixed size
                frame = cv2.resize(frame, thumbnail_size)

                # Convert the frame to a QPixmap
                pixmap = QPixmap.fromImage(
                    QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format.Format_BGR888))

                # Display the thumbnail in the QLabel
                self.thumbnail_label.setPixmap(pixmap)
                self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                QMessageBox.critical(self, "Error", "Failed to read thumbnail from video")

            # Release the video capture object
            cap.release()

    def extract_person(self):
        if not self.file_path.endswith(".mp4"):
            QMessageBox.critical(self, "Error", "Please select a valid MP4 file")
        else:
            # Your code to extract person detection goes here

            # Extract frames
            input_file = self.file_path
            interval = 10  # Extract frames every 5 seconds

            cap = cv2.VideoCapture(input_file)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(frame_rate * interval)

            frame_number = 0
            output_folder = "extracted_images"  # New output folder name
            os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

            # Load the YOLOv4-tiny model
            yolo_config_path = 'yolov4-tiny.cfg'  # Replace with the path to your YOLOv4-tiny.cfg file
            yolo_weights_path = 'yolov4-tiny.weights'  # Replace with the path to your YOLOv4-tiny weights file
            net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

            # Create a list to store the names of files with detected people
            person_detected_files = []

            for file_name in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, file_name))  # Clear previous files

            while frame_number < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    second = frame_number / frame_rate
                    output_file = os.path.join(output_folder,
                                               f"frame_{second:.2f}.jpg")  # Use seconds as part of the filename
                    cv2.imwrite(output_file, frame)

                    # Load and preprocess the input image
                    input_image = frame
                    # Preprocess the input image (resize to uniform dimensions)
                    input_image = cv2.resize(frame, (580, 480))  # Adjust dimensions as need

                    if input_image is not None:
                        height, width, _ = input_image.shape

                        # Create a blob from the image for YOLO input
                        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                        net.setInput(blob)

                        # Get the names of the output layers
                        layer_names = net.getUnconnectedOutLayersNames()

                        # Perform object detection
                        detections = net.forward(layer_names)

                        # Filter detections to get only "person" class
                        detected_people = []

                        for detection in detections:
                            for obj in detection:
                                scores = obj[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]

                                if confidence > 0.5 and class_id == 0:  # Class ID 0 represents "person"
                                    detected_people.append(obj)

                        if detected_people:
                            # If people are detected, save the image with bounding boxes
                            person_detected_files.append(output_file)

                            for obj in detected_people:
                                box = obj[0:4] * np.array([width, height, width, height])
                                (x, y, w, h) = box.astype("int")
                                cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Save the processed image
                            cv2.imwrite(output_file, input_image)

                    frame_number += frame_interval

            cap.release()

            # Add the paths of images with detected persons to the self.detected_images list
            self.detected_images.extend(person_detected_files)

            # Show the detected images in a separate window
            image_viewer = ImageViewer(self.detected_images)
            image_viewer.exec()

            # Print the list of files with detected people
            print("Images with detected people:")
            for file_path in person_detected_files:
                print(file_path)

    def cleanup(self):
        # This function will be called when the application is about to quit
        extracted_images_folder = "extracted_images"
        if os.path.exists(extracted_images_folder):
            # If the folder exists, remove it and its contents
            for file_name in os.listdir(extracted_images_folder):
                file_path = os.path.join(extracted_images_folder, file_name)
                os.remove(file_path)
            os.rmdir(extracted_images_folder)

def main():
    app = QApplication(sys.argv)
    window = PersonDetectionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
