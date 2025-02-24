import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image  # Ensure Pillow is installed: pip install Pillow

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

        # Create a new folder within the project for the copied images.
        copy_folder = os.path.join(os.getcwd(), "copied_images")
        os.makedirs(copy_folder, exist_ok=True)

        # Build the new file name with '_copy' added before the extension.
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

if __name__ == "__main__":
    # Change this to the path of your folder.
    folder_to_watch = r"test_images"

    event_handler = ImageRenameHandler(folder_to_watch)
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)

    print(f"Watching folder: {folder_to_watch}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
