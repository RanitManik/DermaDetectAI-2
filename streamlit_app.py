import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Load labels
with open("label.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

# Define model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load("skin_disease_model.pth", map_location=torch.device("cpu")))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🧴 Skin Disease Classifier")

uploaded_file = st.file_uploader("Upload an image of the skin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs[0], dim=0)

    confidences = {label: float(probs[i]) for i, label in enumerate(labels)}
    top_prediction = labels[torch.argmax(probs).item()]

    st.subheader(f"🩺 Prediction: `{top_prediction}`")

    # Prepare data for bar chart
    df_confidences = pd.DataFrame({
        'Confidence': list(confidences.values())
    }, index=confidences.keys())

    # Display horizontal bar chart
    st.bar_chart(df_confidences, use_container_width=True)
