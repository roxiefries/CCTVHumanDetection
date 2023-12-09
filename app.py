import cv2
import os
import sys
import shutil
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox, QVBoxLayout, \
    QWidget, QSlider, QDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QIcon



os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(os.path.dirname(__file__), 'platforms')


class ImageViewer(QDialog):
    def __init__(self, images):
        super().__init__()

        self.setWindowTitle("Detected Images")
        self.setFixedSize(1000, 900)  # Set a fixed size for the window
        layout = QVBoxLayout()

        self.instruct_label = QLabel("The extracted images are located in your documents folder named 'extracted_images'")
        self.instruct_label.setStyleSheet("color: white; font-size: 20px;")  # Set the color to blue and font size to 16px
        layout.addWidget(self.instruct_label)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.images = images
        self.current_index = 0

        self.show_image()

        self.setLayout(layout)
        self.label_label = QLabel(
            "to navigate through the detected person, just pressed left or right button")
        self.label_label.setStyleSheet(
            "color: white; font-size: 20px;")  # Set the color to blue and font size to 16px
        layout.addWidget(self.label_label)

        self.detected_label = QLabel(
            "If you wish to save the selected image on desired folder, click the save button :)")
        self.detected_label.setStyleSheet(
            "color: white; font-size: 20px;")  # Set the color to blue and font size to 16px
        layout.addWidget(self.detected_label)

        self.image_index_label = QLabel(
            "----")
        self.image_index_label.setStyleSheet(
            "color: white; font-size: 20px;")  # Set the color to blue and font size to 16px
        layout.addWidget(self.image_index_label)

        self.save_button = QPushButton("Save Image")
        self.save_button.setStyleSheet("""
                         background-color: blue;
                         color: white;
                         font-weight: bold;
                         text-transform: uppercase;
                         font-size: 20px;
                         padding: 10px;
                     """)
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

        self.setWindowTitle("PersonDetection App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        # Create a QLabel widget to display the image
        image_label = QLabel()
        layout.addWidget(image_label)


        mp4_label = QLabel("Only MP4 files are allowed")
        mp4_label.setStyleSheet("""
                    font-size: 18px; /* Adjust font size to make it bigger */
                """)

        layout.addWidget(mp4_label)
        # Load the image
        pixmap = QPixmap("/Users/aynnarosepineda/PycharmProjects/CCTVHumanExtraction/logo1.png")  # Replace with your image file path\
        # Set the desired height and width for the image
        new_width = 400  # Replace with your desired width
        new_height = 150  # Replace with your desired height

        # Resize the image to the specified dimensions
        pixmap = pixmap.scaled(new_width, new_height)

        # Create a QLabel and set the pixmap
        image_label = QLabel()

        image_label.setPixmap(pixmap)

        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add the QLabel to the layout
        layout.addWidget(image_label)


        self.browse_button = QPushButton("Upload MP4 File")
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setStyleSheet("""
                  background-color: red;
                  color: white;
                  font-weight: bold;
                  text-transform: uppercase;
                  font-size: 16px;
                  padding: 10px;
              """)
        layout.addWidget(self.browse_button)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: green; font-size: 20px;")  # Set the color to blue and font size to 16px
        layout.addWidget(self.file_label)

        self.thumbnail_label = QLabel()  # QLabel to display the thumbnail
        layout.addWidget(self.thumbnail_label)

        self.extract_button = QPushButton("Extract Person Detected")
        self.extract_button.setStyleSheet("""
                      background-color: lightgreen;
                      color: black;
                      font-weight: bold;
                      text-transform: uppercase;
                      font-size: 16px;
                      padding: 10px;

                  """)

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
            interval = 10  # Extract frames every 10 seconds

            cap = cv2.VideoCapture(input_file)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(frame_rate * interval)

            frame_number = 0
            import os

            # Get the user's home directory
            home_dir = os.path.expanduser('~')

            # Path to the Documents folder
            documents_folder = os.path.join(home_dir, 'Documents')

            # Path to the new output folder inside the Documents folder
            output_folder = os.path.join(documents_folder, 'extracted_images')

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Path to the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Paths to the YOLO files
            yolo_config_path = os.path.join(script_dir, 'yolo', 'yolov4-tiny.cfg')
            yolo_weights_path = os.path.join(script_dir, 'yolo', 'yolov4-tiny.weights')

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
                    input_image = cv2.resize(frame, (800, 580))  # Adjust dimensions as need

                    if input_image is not None:
                        height, width, _ = input_image.shape

                        # Create a blob from the image for YOLO input
                        blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (500, 500), swapRB=True, crop=False)
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

                                if confidence > 0.7 and class_id == 0:  # Class ID 0 represents "person"
                                    detected_people.append(obj)

                        if detected_people:
                            # If people are detected, save the image with bounding boxes
                            person_detected_files.append(output_file)
                            count_detected = 0;
                            for obj in detected_people:
                                box = obj[0:4] * np.array([width, height, width, height])
                                (x, y, w, h) = box.astype("int")
                                # Define how much you want to move the rectangle to the left and up
                                move_x = -10  # Adjust this value as needed for your desired leftward movement
                                move_y = -35  # Adjust this value as needed for your desired upward movement

                                # Update the coordinates of the top-left corner
                                x += move_x
                                y += move_y
                                cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                # Define the text label and its position
                                label = "person"
                                text_position = (
                                x, y - 10)  # Place the label slightly above the top-left corner of the bounding box

                                # Draw the label text on the image
                                cv2.putText(input_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255,255), 2)
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

def main():
    app = QApplication(sys.argv)
    window = PersonDetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
