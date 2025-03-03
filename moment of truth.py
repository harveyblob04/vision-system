import os
import time
import shutil
import threading
import datetime
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image  # Ensure Pillow is installed: pip install Pillow

# -----------------------------
# Watcher 1: Process images from test_images folder
# -----------------------------
class ImageRenameHandler(FileSystemEventHandler):
    def __init__(self, folder):
        self.folder = folder

    def on_created(self, event):
        # Ignore directories.
        if event.is_directory:
            return

        # Wait until the file exists.
        if not self.wait_for_file(event.src_path, timeout=10):
            print(f"File '{event.src_path}' not found after waiting.")
            return

        self.rename_new_image(event.src_path)

    def wait_for_file(self, file_path, timeout=10):
        """Wait until the file exists or until the timeout is reached."""
        start_time = time.time()
        while not os.path.exists(file_path):
            time.sleep(0.5)
            if time.time() - start_time > timeout:
                return False
        return True

    def rename_new_image(self, file_path):
        folder = self.folder
        _, ext = os.path.splitext(file_path)

        # Get all file names in the folder that are already numbers.
        current_numbers = []
        for f in os.listdir(folder):
            name, f_ext = os.path.splitext(f)
            if name.isdigit():
                current_numbers.append(int(name))

        # Determine the next available number.
        next_number = max(current_numbers) + 1 if current_numbers else 1
        new_filename = f"{next_number}{ext}"
        new_filepath = os.path.join(folder, new_filename)

        # Retry loop for renaming, handling both file locks and file disappearance.
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            # Check if the source file still exists.
            if not os.path.exists(file_path):
                print(f"Source file '{file_path}' not found. Retrying (attempt {attempt+1}/{max_attempts})...")
                attempt += 1
                time.sleep(1)
                continue
            try:
                os.rename(file_path, new_filepath)
                print(f"Renamed '{file_path}' to '{new_filepath}'")
                break  # Exit loop if rename is successful.
            except OSError as e:
                if getattr(e, 'winerror', None) == 32:
                    attempt += 1
                    print(f"File is in use, retrying rename (attempt {attempt}/{max_attempts})...")
                    time.sleep(1)
                else:
                    print(f"Error renaming file: {e}")
                    return
        else:
            print(f"Error renaming file: file remains inaccessible after {max_attempts} attempts.")
            return

        # Create a new folder for the copied images.
        copy_folder = os.path.join(os.getcwd(), "copied_images")
        os.makedirs(copy_folder, exist_ok=True)

        # Build the new file name with '_grayscale' added before the extension.
        name_only, ext = os.path.splitext(new_filename)
        copy_filename = f"{name_only}_grayscale{ext}"
        copy_filepath = os.path.join(copy_folder, copy_filename)

        try:
            shutil.copy2(new_filepath, copy_filepath)
            print(f"Copied '{new_filepath}' to '{copy_filepath}'")
        except Exception as e:
            print(f"Error copying file: {e}")
            return

        # Convert the copied image to grayscale.
        try:
            self.convert_to_grayscale(copy_filepath)
        except Exception as e:
            print(f"Error converting image to grayscale: {e}")

    def convert_to_grayscale(self, image_path):
        """Open an image, convert it to grayscale, and save it."""
        try:
            img = Image.open(image_path)
            grayscale = img.convert("L")  # 'L' mode is for grayscale.
            grayscale.save(image_path)
            print(f"Converted '{image_path}' to grayscale")
        except Exception as e:
            print(f"Error during grayscale conversion: {e}")

def start_image_watcher():
    folder_to_watch = os.path.join(os.getcwd(), "test_images")
    os.makedirs(folder_to_watch, exist_ok=True)
    event_handler = ImageRenameHandler(folder_to_watch)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    print(f"Starting watcher for new images in: {folder_to_watch}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# -----------------------------
# Watcher 2: Process grayscale images from copied_images folder
# -----------------------------
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
        cv2.waitKey(500)  # show for a brief moment

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
        cv2.waitKey(500)
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

def start_copied_images_watcher():
    folder_to_watch = os.path.join(os.getcwd(), "copied_images")
    os.makedirs(folder_to_watch, exist_ok=True)
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_to_watch, recursive=False)
    print(f"Starting watcher for grayscale images in: {folder_to_watch}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# -----------------------------
# Webcam Capture: Capture images on key press and save to test_images folder.
# -----------------------------
def start_webcam_capture():
    output_folder = os.path.join(os.getcwd(), "test_images")
    os.makedirs(output_folder, exist_ok=True)
    # Open the default camera (change index if necessary)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'p' to capture an image, or 'q' to quit webcam capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting webcam capture...")
            break

        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            # Generate a filename with the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"capture_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Image saved as: {filename}")
        elif key == ord('q'):
            print("Exiting webcam capture...")
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Main function: start threads for each functionality.
# -----------------------------
def main():
    # Create threads for each subprogram.
    watcher_thread = threading.Thread(target=start_image_watcher, daemon=True)
    processor_thread = threading.Thread(target=start_copied_images_watcher, daemon=True)
    webcam_thread = threading.Thread(target=start_webcam_capture, daemon=True)

    # Start the threads.
    watcher_thread.start()
    processor_thread.start()
    webcam_thread.start()

    # Keep the main thread alive until webcam capture ends.
    webcam_thread.join()
    print("Webcam capture ended. Exiting program...")

if __name__ == "__main__":
    main()
