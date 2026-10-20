from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles











import tensorflow as tf
import numpy as np
import cv2
import json
import tensorflow as tf
import numpy as np
import cv2
import json


app = FastAPI(
    title="Bharatiya Natya Mudra Recognition API"
)


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

MODEL_PATH = "mudra_cnn_final.keras"
MAPPING_PATH = "class_mapping.json"


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# templates = Jinja2Templates(
#     directory="templates"
# )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)

print("Class mapping loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid image")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


templates = Jinja2Templates(
    directory="templates"
)

# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    return {
        "status": "running",
        "service": "Bharatiya Natya Mudra Recognition API"
    }


# ============================================================
# MEDIAPIPE MODEL FOR BROWSER
# ============================================================

@app.get("/hand_landmarker.task")
async def hand_landmarker_model():

    return FileResponse(
        "hand_landmarker.task",
        media_type="application/octet-stream"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    try:

        input_image = preprocess_image(
            image_bytes
        )

    except ValueError:

        return {
            "mudra": None,
            "confidence": 0,
            "class_id": None,
            "error": "Invalid image"
        }

    probabilities = model.predict(
        input_image,
        verbose=0
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    predicted_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_id]
    )

    predicted_mudra = class_mapping[
        str(predicted_id)
    ]

    return {
        "mudra": predicted_mudra,
        "confidence": round(
            confidence * 100,
            2
        ),
        "class_id": predicted_id
    }