import cv2
import numpy as np
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Define a handler for new files in the folder
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Only process files (not directories) and check if "grayscale" is in the file name
        if not event.is_directory and "grayscale" in os.path.basename(event.src_path):
            print(f"New grayscale image detected: {event.src_path}")
            # Add a small delay to ensure the file is fully written
            time.sleep(0.5)
            process_image(event.src_path)


def process_image(filepath):
    # Check if file exists and is readable
    if not os.path.exists(filepath):
        print(f"Error: File does not exist: {filepath}")
        return

    if not os.access(filepath, os.R_OK):
        print(f"Error: File is not readable: {filepath}")
        return

    try:
        # Print file size and last modified time for debugging
        file_size = os.path.getsize(filepath)
        file_time = os.path.getmtime(filepath)
        print(f"File size: {file_size} bytes")
        print(f"Last modified: {time.ctime(file_time)}")

        # Read the image in grayscale
        print(f"Attempting to read image: {filepath}")
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


        if image is None:
            print(f"Error: OpenCV failed to read image {filepath}")
            return

        print(f"Successfully read image. Shape: {image.shape}")

        # Thresholding
        ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

        # Zero out the border pixels to prevent detecting the image edge as a contour
        thresh[0, :] = 0
        thresh[-1, :] = 0
        thresh[:, 0] = 0
        thresh[:, -1] = 0

        # Now find external contours in the modified thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edged = cv2.Canny(image, 30, 200)

        cv2.imshow('edged', edged)
        cv2.waitKey(0)

        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        cv2.imshow('contours', image)
        cv2.waitKey(0)


        print(contours)
        print(hierarchy)
        if not contours:
            print(f"No contours found in {filepath}")
            return

        # Assume the largest contour corresponds to the sheet
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments of the largest contour to find the center
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            print(f"Contour area is zero for {filepath}")
            return
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(f"Center of sheet in {filepath}: ({cX}, {cY})")

        # Create a text file to save the coordinates with a new naming convention
        base_name = os.path.basename(filepath)
        number = base_name.split('_')[0]  # Gets the number before the first underscore
        txt_filename = os.path.join(os.path.dirname(filepath), f"{number}_coords.txt")

        with open(txt_filename, "w") as f:
            f.write(f"{cX}, {cY}")
        print(f"Coordinates saved to {txt_filename}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")


if __name__ == "__main__":
    # Set the folder path to be watched with absolute path
    folder_to_watch = r"C:\Users\harve\PycharmProjects\PythonProject\copied_images"

    # Verify the folder exists
    if not os.path.exists(folder_to_watch):
        print(f"Error: Folder {folder_to_watch} does not exist!")
        exit(1)

    print(f"Starting to watch folder: {folder_to_watch}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Folder exists: {os.path.exists(folder_to_watch)}")
    print(f"Folder is readable: {os.access(folder_to_watch, os.R_OK)}")

    # Set up the observer and event handler
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_to_watch, recursive=False)

    observer.start()

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()