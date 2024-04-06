import torch
import gradio as gr
import cv2
import numpy as np
from mtcnn import MTCNN
from efficientnet_pytorch import EfficientNet

detector = MTCNN()

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

def predict_image(image):
    model = EfficientNet.from_name('efficientnet-b2', num_classes=2)
    model.load_state_dict(torch.load('best.pkl', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        image_tensor = load(image)
        if image_tensor is not None:
            outputs = model(image_tensor)
            _, preds = torch.max(outputs.data, 1)
            return "Prediction: {}".format("Deepfake" if preds.item() == 0 else "Real")
        else:
            return "No face detected"

iface = gr.Interface(predict_image, inputs="image", outputs="text", title="Deepfake Detection")
iface.launch(share=True, debug=True)
