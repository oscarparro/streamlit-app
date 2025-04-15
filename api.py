# api.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import face_recognition
import io
from PIL import Image
import base64

app = FastAPI()

# Habilitamos CORS para permitir peticiones desde la app (por ejemplo, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionarios globales para almacenar los datos de las caras registradas
# registered_faces: clave = nombre, valor = encoding facial (como np.array)
# registered_face_images: clave = nombre, valor = imagen original codificada en base64
registered_faces = {}
registered_face_images = {}

@app.post("/register_face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """
    Recibe el nombre y la imagen que debe contener UN único rostro.
    Se extrae el encoding y se almacena, junto con la imagen codificada en base64, 
    para poder consultarla posteriormente.
    """
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception as e:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    # Si la imagen tiene 4 canales (p.ej. PNG con alfa), la convertimos a RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
     
    # Extraer el encoding facial
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return JSONResponse(content={"error": "No se detectó rostro en la imagen."}, status_code=400)
    if len(face_encodings) > 1:
        return JSONResponse(content={"error": "Se detectaron más de un rostro. Registra únicamente una cara."}, status_code=400)
    
    encoding = face_encodings[0]
    # Almacenamos el encoding
    registered_faces[name] = encoding
    # Convertimos la imagen original a base64 para poder visualizarla luego
    encoded_image = base64.b64encode(contents).decode("utf-8")
    registered_face_images[name] = encoded_image

    return {"message": f"Rostro de '{name}' registrado correctamente."}

@app.post("/identify_face")
async def identify_face(file: UploadFile = File(...)):
    """
    Recibe una imagen (archivo) con un rostro, extrae su encoding y lo compara con
    los rostros almacenados. Si hay coincidencia, devuelve el nombre; si no, 'Desconocido'.
    """
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
    except Exception as e:
        return JSONResponse(content={"error": "No se pudo leer la imagen."}, status_code=400)
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return JSONResponse(content={"error": "No se detectó rostro en la imagen."}, status_code=400)
    
    # Suponemos que en la imagen se buscará identificar un único rostro
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
    return {"name": best_match, "distance": resultados[best_match]}

@app.get("/registered_faces")
async def get_registered_faces():
    """
    Devuelve una lista de las caras registradas.
    Por cada persona se retorna:
      - nombre
      - imagen: cadena en base64 que permite visualizar el rostro registrado
    """
    data = []
    for name, img_base64 in registered_face_images.items():
        data.append({"name": name, "image": img_base64})
    return data
