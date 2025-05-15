import cv2
import numpy as np
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Global counters for image and text files
image_counter = 0
text_counter = 0

def get_new_filename(original_path):
    """
    Returns a new file name based on the file type.
    If the file is an image, rename to image_<number>.<ext>.
    If it's a text file, rename to coords_<number>.txt.
    For other file types, the original file name is retained.
    """
    global image_counter, text_counter
    directory, original_name = os.path.split(original_path)
    name, ext = os.path.splitext(original_name)
    ext_lower = ext.lower()

    if ext_lower in IMAGE_EXTENSIONS:
        image_counter += 1
        new_filename = f"image_{image_counter}{ext_lower}"
    elif ext_lower == ".txt":
        new_filename = f"coords_{image_counter}.txt"
    else:
        # For other file types, keep the original file name
        new_filename = original_name

    return os.path.join(directory, new_filename)

# Define a handler for new files in the folder
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Only process files (ignore directories)
        if not event.is_directory:
            # Wait briefly to ensure the file is fully written
            time.sleep(0.5)
            new_filepath = get_new_filename(event.src_path)
            try:
                os.rename(event.src_path, new_filepath)
                print(f"Renamed '{event.src_path}' to '{new_filepath}'")
            except Exception as e:
                print(f"Error renaming file {event.src_path}: {e}")
                return

            # If the file is an image, process it; ignore text files or others
            _, ext = os.path.splitext(new_filepath)
            if ext.lower() in IMAGE_EXTENSIONS:
                process_image(new_filepath)
            else:
                print(f"Ignoring non-image file: {new_filepath}")

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

        # Thresholding to create a binary image (black and white)
        ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

        # Save the thresholded (black and white) image copy
        base_name, ext = os.path.splitext(os.path.basename(filepath))
        bw_filename = os.path.join(os.path.dirname(filepath), f"{base_name}_bw{ext}")
        cv2.imwrite(bw_filename, thresh)
        print(f"Black and white image saved as {bw_filename}")

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

        # Save the coordinates to a text file based on the image file name
        name_without_ext = os.path.splitext(os.path.basename(filepath))[0]
        txt_filename = os.path.join(os.path.dirname(filepath), f"{name_without_ext}_coords.txt")

        with open(txt_filename, "w") as f:
            for center in rectangle_centers:
                f.write(f"{center[0]}, {center[1]}\n")
        print(f"Coordinates saved to {txt_filename}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    # Set the folder path to be watched (using an absolute path)
    folder_to_watch = os.path.join(os.getcwd(), "copied_images")

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
