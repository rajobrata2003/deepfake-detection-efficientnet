import torch
import gradio as gr
import cv2
import numpy as np
from mtcnn import MTCNN
from efficientnet_pytorch import EfficientNet
import random

detector = MTCNN()

def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indexes = random.sample(range(frame_count), min(num_frames, frame_count))
    frames = []
    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def load(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    if faces:
        face = faces[0]
        x, y, w, h = face['box']
        cropped_face = image[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (150, 150))
        cropped_face = cropped_face.astype(np.float32) / 255.0
        cropped_face = np.transpose(cropped_face, (2, 0, 1))
        image_tensor = torch.from_numpy(cropped_face).float()
        return image_tensor.unsqueeze(0)
    else:
        return None

def predict_video(video):
    frames = extract_frames(video)
    real_count = 0  # Counter for frames classified as "Real"
    deepfake_count = 0  # Counter for frames classified as "Deepfake"
    total_frames = len(frames)
    model = EfficientNet.from_name('efficientnet-b2', num_classes=2)
    model.load_state_dict(torch.load('best.pkl', map_location=torch.device('cpu')))
    model.eval()

    for frame in frames:
        with torch.no_grad():
            image_tensor = load(frame)
            if image_tensor is not None:
                outputs = model(image_tensor)
                _, preds = torch.max(outputs.data, 1)
                if preds.item() == 0:  # If frame classified as "Deepfake"
                    deepfake_count += 1
                else:  # If frame classified as "Real"
                    real_count += 1

    confidence_deepfake = (deepfake_count / total_frames) * 100
    confidence_real = (real_count / total_frames) * 100

    if confidence_deepfake > 70:
        return "Prediction: Deepfake, Confidence: {:.2f}%".format(confidence_deepfake)
    elif confidence_real > 70:
        return "Prediction: Real, Confidence: {:.2f}%".format(confidence_real)
    else:
        return "Prediction: Uncertain"

iface = gr.Interface(predict_video, inputs="video", outputs="text", title="Deepfake Detection on Video")
iface.launch(share=True, debug=True)

