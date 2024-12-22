import os
import joblib
import numpy as np
import torch
from torchvision import models, transforms
import gradio as gr

# Load the custom model
model_path = "./models/random_forest_resnet.pth"
model = joblib.load(model_path)

# Load ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
resnet.eval()

# Define the labels for your custom model
labels = ["Colon Cancer", "Not Colon Cancer"]

def extract_features(image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image).numpy().flatten().reshape(1, -1)
    return features

def predict(inp):
    threshold = 0.75  # Set a higher threshold for confidence
    features = extract_features(inp)
    prediction = model.predict_proba(features)[0]
    max_confidence = max(prediction)
    if max_confidence < threshold:
        return {"Uncertain": 1.0}
    confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences

example_images = ["./examples/aca-example.jpeg", "./examples/n-example.jpeg"]

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             examples=example_images).launch(share=True)