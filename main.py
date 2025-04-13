from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import io

app = FastAPI()

# Enable CORS for your Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the model structure
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)  # 6 output classes

# 2. Load the trained weights
model.load_state_dict(torch.load("skin_disease_model.pth", map_location=torch.device("cpu")))
model.eval()

# 3. Load labels
with open("label.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

# 4. Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs[0], dim=0)

    confidences = {label: float(probs[i]) for i, label in enumerate(labels)}
    top_prediction = labels[torch.argmax(probs).item()]

    return {
        "prediction": top_prediction,
        "confidences": confidences
    }

