import os
import glob
import cv2
import time
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import numpy as np
from model.ResNeXt import resnext50
from PIL import Image

# Clear the results/preprocessing_test/ folder
folder_path = "./results/preprocessing_test/"
files = glob.glob(os.path.join(folder_path, "image*.jpg"))
for f in files:
    os.remove(f)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnext50().to(device)
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load the mean and std for normalization
df = pd.read_csv("./data/test_with_emotions.csv")
pixel_arrays = np.array([np.array(row.split(), dtype=np.uint8).reshape(48, 48) for row in df['pixels']])
all_pixels = pixel_arrays.ravel()
mean = all_pixels.mean() / 255.0
std = all_pixels.std() / 255.0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

# Define the duration of the video
duration = 10  # seconds

# Set up the capture
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize the VideoWriter object for grayscale images
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
out = cv2.VideoWriter('./results/output.mp4', fourcc, 20.0, (48, 48), False)  # False for grayscale

# Initialize the MTCNN face detector with increased confidence thresholds
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.7, 0.8, 0.8])

start_time = time.time()
frame_count = 0  # Initialize frame count for naming saved images

# Loop to capture and process frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB (MTCNN expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect faces
    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x, y, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # Extract the face
            face_img = frame_rgb[y:y2, x:x2]
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            # Resize the face image to 48x48
            face_resized = cv2.resize(face_gray, (48, 48))

            # Save the grayscale face image
            cv2.imwrite(f"./results/preprocessing_test/image{frame_count}.jpg", face_resized)

            # Convert the face to a PIL image
            face_pil = Image.fromarray(face_resized)

            # Convert the face to a tensor and apply transformations
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Predict the facial expression
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)

            # Map the predicted label to the corresponding emotion
            emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
            predicted_emotion = emotion_map[predicted.item()]

            # Draw a rectangle around the face and display the predicted emotion
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Write the grayscale face image to the video file
            out.write(face_resized)

    frame_count += 1

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Stop after the specified duration
    if time.time() - start_time > duration:
        break

# When everything is done, release the capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
