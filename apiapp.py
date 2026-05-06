from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import os
import gdown

# -------------------------
# DOWNLOAD MODEL
# -------------------------
MODEL_PATH = "resnet50A2.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=18g7OAIjxBnSos-rOBuSFWIvtYsU_pnG5"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# FASTAPI
# -------------------------
app = FastAPI()

# -------------------------
# CLASS NAMES
# -------------------------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cpu")

# -------------------------
# LOAD MODEL
# -------------------------
model = models.resnet50(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# ROOT
# -------------------------
@app.get("/")
def home():

    return {
        "message": "Brain Tumor API Running"
    }

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = F.softmax(
            outputs,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[0][pred].item() * 100

    return {

        "prediction": class_names[pred],

        "confidence": confidence,

        "probabilities": [

            p.item() * 100

            for p in probs[0]
        ]
    }