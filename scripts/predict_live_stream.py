import sys
import os
import glob
import cv2
import time
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Clear the results/preprocessing_test/ folder
folder_path = "./results/preprocessing_test/"
files = glob.glob(os.path.join(folder_path, "image*.png"))
for f in files:
    os.remove(f)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./model/my_own_model.pkl', map_location=device)
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

# Use the input video file for testing
# Uncomment the following lines to use the input video file and comment the next lines
# cap = cv2.VideoCapture('./results/input_video.mp4')
# if not cap.isOpened():
#     print("Cannot open video file")
#     exit()
# print("Reading video stream ...")


cap = cv2.VideoCapture(0)
use_camera = cap.isOpened()

if not use_camera:
    cap = cv2.VideoCapture('./results/input_video.mp4')
    print("Reading video stream ...")


# Initialize the VideoWriter object for grayscale images
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
fps = 30.0  # Set the desired fps value
out_grayscale = cv2.VideoWriter('./results/output.mp4', fourcc, fps, (48, 48), False)  # False for grayscale

# Initialize the VideoWriter object for the original frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_original = cv2.VideoWriter('./results/input_video_.mp4', fourcc, fps, (frame_width, frame_height), True)  # True for color

# Load pre-trained model and configuration file for face detection
model_file = "deploy.prototxt"  # Path to the .prototxt file
config_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"  # Path to the .caffemodel file
net = cv2.dnn.readNetFromCaffe(model_file, config_file)

start_time = time.time()
frame_count = 0  # Initialize frame count for naming saved images
image_count = 0  # Initialize image count for saving images sequentially

# Loop to capture and process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    original_frame = frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            face_img = frame[y:y2, x:x2]
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))

            cv2.imwrite(f"./results/preprocessing_test/image{image_count}.png", face_resized)
            image_count += 1

            face_pil = Image.fromarray(face_resized)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(face_tensor)
                _, predicted = torch.max(outputs.data, 1)

            emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
            predicted_emotion = emotion_map[predicted.item()]

            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            elapsed_time = time.time() - start_time
            frames_elapsed = elapsed_time * fps
            frames_to_add = int(frames_elapsed - frame_count)

            for _ in range(frames_to_add):
                out_grayscale.write(face_resized)
                out_original.write(original_frame)
                frame_count += 1

            current_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
            print(f"Preprocessing ...\n{current_time}s : {predicted_emotion} , {confidence:.2f}%")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out_grayscale.release()
out_original.release()
cv2.destroyAllWindows()
