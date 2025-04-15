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

app = FastAPI()

# Habilitar CORS para permitir peticiones desde la app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#############################
# Detector DNN para identificación
#############################
prototxt_path = "model/deploy.prototxt"
caffe_model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# Cargar el modelo de detección de emociones desde Hugging Face
emotion_model_name = "oscarparro/emotion_detection_vit"
emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_name)

# Transformaciones para preprocesar la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar la imagen al tamaño esperado por ViT
    transforms.ToTensor(),         # Convertir la imagen a un tensor
    transforms.Normalize([0.5], [0.5])  # Normalizar los valores de píxeles
])

def detect_faces_dnn_api(image, net, conf_threshold=0.7):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def predict_emotion(face_img):
    """
    Predice la emoción en una imagen de rostro usando el modelo de Hugging Face.
    """
    # Convertir la imagen a RGB si es necesario
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_img)

    # Aplicar las transformaciones
    input_tensor = transform(pil_image).unsqueeze(0)  # Añadir una dimensión batch

    # Realizar la predicción
    with torch.no_grad():
        outputs = emotion_model(input_tensor)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    emotion = torch.argmax(probs, dim=1).item()

    # Decodificar la etiqueta de la emoción
    emotion_label = emotion_model.config.id2label[emotion]
    return emotion_label

# Diccionario global para almacenar los encodings de rostros registrados: nombre -> encoding facial
registered_faces = {}

###########################################
# Endpoint de Registro (usa detección clásica)
###########################################
@app.post("/register_face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """
    Registra un rostro usando detección clásica basada en HOG para obtener la cara.
    Se recorta la cara y se calcula el encoding sin usar known_face_locations.
    """
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return JSONResponse(content={"error": "No se detectó rostro en la imagen."}, status_code=400)
    if len(face_locations) > 1:
        return JSONResponse(content={"error": "Se detectaron múltiples rostros. Registra únicamente uno."}, status_code=400)
    
    # face_locations devuelve (top, right, bottom, left)
    top, right, bottom, left = face_locations[0]
    face_img = image[top:bottom, left:right]
    face_encodings = face_recognition.face_encodings(face_img)
    if len(face_encodings) == 0:
        return JSONResponse(content={"error": "No se pudo extraer el encoding facial."}, status_code=400)
    encoding = face_encodings[0]
    registered_faces[name] = encoding
    return {"message": f"Rostro de '{name}' registrado correctamente."}

###########################################
# Endpoint de Identificación (usa detector DNN)
###########################################
@app.post("/identify_face")
async def identify_face(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    boxes = detect_faces_dnn_api(image, net, conf_threshold=0.7)
    if len(boxes) == 0:
        return JSONResponse(content={"error": "No se detectó rostro en la imagen."}, status_code=400)
    
    (x, y, w, h) = boxes[0]
    # Convertir el bounding box (x, y, w, h) a (top, right, bottom, left)
    face_location = (y, x + w, y + h, x)
    # En lugar de recortar la imagen, se pasa la imagen completa con la ubicación indicada
    face_encodings = face_recognition.face_encodings(image, known_face_locations=[face_location])
    if len(face_encodings) == 0:
        return JSONResponse(content={"error": "No se pudo extraer encoding facial."}, status_code=400)
    encoding = face_encodings[0]
    
    resultados = {}
    for name, reg_encoding in registered_faces.items():
        match = face_recognition.compare_faces([reg_encoding], encoding, tolerance=0.5)
        distance = face_recognition.face_distance([reg_encoding], encoding)
        if match[0]:
            resultados[name] = float(distance[0])
    if not resultados:
        return {"name": "Desconocido"}
    best_match = min(resultados, key=resultados.get)

    # Recortar la cara para predecir la emoción
    face_img = image[y:y + h, x:x + w]
    emotion = predict_emotion(face_img)

    return {"name": best_match, "distance": resultados[best_match], "emotion": emotion}