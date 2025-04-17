from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageClassification
from torchvision import transforms
import torch
import numpy as np
import cv2
import face_recognition
import io
from PIL import Image

import os
import json
import uuid
import random
import base64
from datetime import datetime

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

#############################
# Detector DNN para identificación
#############################
prototxt_path = "model/deploy.prototxt"
caffe_model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# Emoción
emotion_model_name = "oscarparro/emotion_detection_vit"
emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def detect_faces_dnn_api(image, net, conf_threshold=0.7):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def predict_emotion(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_img)
    input_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = emotion_model(input_tensor)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()
    return emotion_model.config.id2label[idx]

#############################
# Persistencia en JSON
#############################
REGISTERED_FACES_FILE = "data/registered_faces.json"

def load_registered_faces():
    if not os.path.exists(REGISTERED_FACES_FILE):
        return {}
    try:
        with open(REGISTERED_FACES_FILE, "r") as f:
            data = json.load(f)
        # reconvertir encoding a ndarray
        for rec in data.values():
            rec["encoding"] = np.array(rec["encoding"])
        return data
    except (json.JSONDecodeError, ValueError):
        return {}

def save_registered_faces():
    serial = {}
    for fid, rec in registered_faces.items():
        serial[fid] = {
            "name": rec["name"],
            "encoding": rec["encoding"].tolist(),
            "color": rec["color"],
            "time": rec["time"],
            "image_b64": rec["image_b64"]
        }
    with open(REGISTERED_FACES_FILE, "w") as f:
        json.dump(serial, f, indent=2)

# Cargamos los registros al iniciar
registered_faces = load_registered_faces()

###########################################
# Endpoint de Registro
###########################################
@app.post("/register_face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return JSONResponse(content={"error": "Debe haber exactamente un rostro."}, status_code=400)

    top, right, bottom, left = face_locations[0]
    face_img = image[top:bottom, left:right]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(face_rgb)
    if not encs:
        return JSONResponse(content={"error": "No se pudo extraer el encoding facial."}, status_code=400)
    
    fid = str(uuid.uuid4())
    encoding = encs[0]
    color = [random.randint(0, 255) for _ in range(3)]
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # face_img está en RGB, OpenCV espera BGR → convertimos
    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', face_bgr)
    image_b64 = base64.b64encode(buf).decode()

    registered_faces[fid] = {
        "name": name,
        "encoding": encoding,
        "color": color,
        "time": time_str,
        "image_b64": image_b64
    }
    save_registered_faces()
    return {
        "message": f"Rostro de '{name}' registrado correctamente.",
        "record": {
            "id": fid,
            "name": name,
            "color": color,
            "time": time_str,
            "image_b64": image_b64
        }
    }

###########################################
# Listar registros
###########################################
@app.get("/list_faces")
async def list_faces():
    return [
        {
            "id": fid,
            "name": rec["name"],
            "color": rec["color"],
            "time": rec["time"],
            "image_b64": rec["image_b64"]
        }
        for fid, rec in registered_faces.items()
    ]

###########################################
# Eliminar registro
###########################################
@app.post("/delete_face")
async def delete_face(id: str = Form(...)):
    if id in registered_faces:
        del registered_faces[id]
        save_registered_faces()
        return {"message": f"Registro {id} eliminado."}
    return JSONResponse(content={"error": "ID no encontrado."}, status_code=404)

###########################################
# Endpoint de Identificación con Modos
###########################################
@app.post("/identify_face")
async def identify_face(
    file: UploadFile = File(...),
    mode: str = Form("Deteccion completa")
):
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    mode_lower = mode.lower().strip()
    if mode_lower == "no hacer nada":
        return {"message": "Modo inactivo. Se muestra solo la imagen de la cámara."}
    if mode_lower == "deteccion emociones":
        emo = predict_emotion(image)
        return {"name": "N/A", "emotion": emo}

    # deteccion de rostros para los otros modos
    boxes = detect_faces_dnn_api(image, net, conf_threshold=0.7)
    if not boxes:
        return JSONResponse(content={"error": "No se detectó rostro en la imagen."}, status_code=400)
    x, y, w, h = boxes[0]
    face = image[y:y+h, x:x+w]
    face_encs = face_recognition.face_encodings(image, known_face_locations=[(y, x+w, y+h, x)])
    if not face_encs:
        return JSONResponse(content={"error": "No se pudo extraer encoding facial."}, status_code=400)
    enc = face_encs[0]

    # reconocimiento
    resultados = {}
    for rec in registered_faces.values():
        dist = face_recognition.face_distance([rec["encoding"]], enc)[0]
        if face_recognition.compare_faces([rec["encoding"]], enc, tolerance=0.5)[0]:
            resultados[rec["name"]] = float(dist)
    if resultados:
        best = min(resultados, key=resultados.get)
        distance_val = resultados[best]
    else:
        best, distance_val = "Desconocido", None

    resp = {"name": best, "distance": distance_val}
    if mode_lower == "deteccion completa":
        resp["emotion"] = predict_emotion(face)
    return resp
