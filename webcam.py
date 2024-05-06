import cv2
import time
import pandas as pd
from facenet_pytorch import MTCNN
import torch

# Define the duration of the video
duration = 5  # seconds

# Set up the capture
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize the MTCNN face detector
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

# Initialize a list to store the pixel values of all frames
frame_list = []

start_time = time.time()

# Loop to capture and process frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB (MTCNN expects RGB)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect faces
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # Extract the face
            face_img = frame[y:y2, x:x2]
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            # Resize the face image to 48x48
            face_resized = cv2.resize(face_gray, (48, 48))

            # Flatten the 48x48 image to a 1D array of 2304 elements
            flat_image = face_resized.flatten()

            # Append the flattened image to the list
            frame_list.append(pd.Series(flat_image))

            # Draw a rectangle around the face in the original frame
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Delay to capture at 1 fps
    time.sleep(0.25)

    # Stop after the specified duration
    if time.time() - start_time > duration:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# Convert the list of Series to a DataFrame
all_frames = pd.concat(frame_list, axis=1).T

# Save the DataFrame to a single CSV file
all_frames.to_csv('all_frames.csv', header=False, index=False)
