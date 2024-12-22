from PIL import Image
from torchvision import transforms, models
import torch
import gradio as gr
import os
import joblib
import numpy as np

# Load the custom model
model_path = "./models/random_forest_vgg16.pth"
model = joblib.load(model_path)

# Load VGG16 model for feature extraction
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])  # Remove the last layer
vgg16.eval()

# Define the labels for your custom model
labels = ["Colon Cancer", "Not Colon Cancer"]  # Update these labels according to your model's classes

def extract_features(image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = vgg16(image).numpy().flatten().reshape(1, -1)
    return features

def predict(inp):
    features = extract_features(inp)
    prediction = model.predict_proba(features)[0]
    confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences

example_images = ["./examples/aca-example.jpeg", "./examples/n-example.jpeg"]

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             examples=example_images).launch(share=True)