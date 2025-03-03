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
        # Debug: File size and modification time
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

        # Thresholding to create a binary image
        ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

        # Zero out the border pixels to prevent edge detection on the image border
        thresh[0, :] = 0
        thresh[-1, :] = 0
        thresh[:, 0] = 0
        thresh[:, -1] = 0

        # Find external contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Display the edged image for debugging
        edged = cv2.Canny(image, 30, 200)
        cv2.imshow('edged', edged)
        cv2.waitKey(0)

        # Convert grayscale image to BGR so we can draw colored contours and centers
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rectangle_centers = []

        # Process each contour to check if it is a rectangle and compute its center
        for cnt in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # If the polygon has 4 vertices, consider it as a rectangle
            if len(approx) == 4:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                rectangle_centers.append((cX, cY))
                # Draw the rectangle and its center on the image
                cv2.drawContours(image_color, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(image_color, (cX, cY), 5, (0, 0, 255), -1)

        cv2.imshow('rectangles', image_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not rectangle_centers:
            print(f"No rectangles found in {filepath}")
            return

        print("Detected rectangle centers:", rectangle_centers)

        # Create a text file to save the coordinates.
        # The file naming convention uses the first part of the original file name.
        base_name = os.path.basename(filepath)
        number = base_name.split('_')[0]  # Gets the number before the first underscore
        txt_filename = os.path.join(os.path.dirname(filepath), f"{number}_coords.txt")

        with open(txt_filename, "w") as f:
            for center in rectangle_centers:
                f.write(f"{center[0]}, {center[1]}\n")
        print(f"Coordinates saved to {txt_filename}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    # Set the folder path to be watched (using an absolute path)
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
