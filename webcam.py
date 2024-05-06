import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt


# Define the duration of the video
duration = 20  # seconds

# Set up the capture
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

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

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Flatten the 48x48 image to a 1D array of 2304 elements
    flat_image = resized.flatten()

    # Append the flattened image to the list
    frame_list.append(pd.Series(flat_image))
    
    # Display the resulting frame
    cv2.imshow('frame', resized)
    if cv2.waitKey(1) == ord('q'):
        break

    # Delay to capture at 1 fps
    time.sleep(0.5)

    # Stop after 20 seconds
    if time.time() - start_time > duration:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# Convert the list of Series to a DataFrame
all_frames = pd.concat(frame_list, axis=1).T

# Save the DataFrame to a single CSV file
all_frames.to_csv('all_frames.csv', header=False, index=False)

