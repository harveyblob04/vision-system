import cv2
import os
import datetime


def main():
    # Define the folder to save captured images
    output_folder = "captured_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'p' to capture an image, or 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Display the current frame in a window
        cv2.imshow('Camera Feed', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            # Generate a filename with the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"capture_{timestamp}.png")

            # Save the captured image
            cv2.imwrite(filename, frame)
            print(f"Image saved as: {filename}")
        elif key == ord('q'):
            # Exit the loop when 'q' is pressed
            print("Exiting...")
            break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
