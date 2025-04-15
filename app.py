import streamlit as st
import cv2
import numpy as np
import time
import requests
import random
import base64
import datetime

#########################
# Configuración y funciones de API
#########################

# URL de la API (asegúrate de que esté corriendo, por ejemplo, con "uvicorn api:app --reload")
API_URL = "http://127.0.0.1:8000"

# Ruta y carga del clasificador Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Parámetros para detección con Haar Cascade
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
DETECTION_MIN_SIZE = (30, 30)

# Color para "Desconocido" (BGR)
UNKNOWN_COLOR = (0, 255, 0)

def register_face_api(face_img, name):
    """Envía la imagen (recortada de la cara) y el nombre a la API para su registro."""
    _, img_encoded = cv2.imencode('.jpg', face_img)
    files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'name': name}
    response = requests.post(API_URL + "/register_face", files=files, data=data)
    return response.json()

def identify_face_api(face_img):
    """Envía la imagen (recortada de la cara) a la API para identificarla."""
    _, img_encoded = cv2.imencode('.jpg', face_img)
    files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(API_URL + "/identify_face", files=files)
        return response.json()
    except Exception:
        return {"name": "Error"}

def generate_random_color():
    """Devuelve un color aleatorio (tupla BGR)."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

#########################
# Estado compartido (session_state)
#########################

if "registrations" not in st.session_state:
    st.session_state.registrations = []  # Lista de diccionarios con registros: fecha, nombre, color y imagen
if "registered_colors" not in st.session_state:
    st.session_state.registered_colors = {}
if "identification_active" not in st.session_state:
    st.session_state.identification_active = False

#########################
# Navegación: Se elige la vista en la barra lateral
#########################

view = st.sidebar.radio("Navegar", ["Registro", "Identificación", "Historial"])
st.session_state.selected_view = view  # Guardamos la vista seleccionada

#########################
# Vista 1: Registro
#########################
if view == "Registro":
    st.title("Registro de Rostro")
    st.write("Captura una foto y regístrate.")
    # Usamos el componente nativo para cámara (toma una foto)
    img_file = st.camera_input("Captura tu rostro")
    user_name = st.text_input("Nombre:")

    if st.button("Registrar"):
        if img_file is None:
            st.error("Por favor, toma una foto primero.")
        elif user_name.strip() == "":
            st.error("Ingresa un nombre.")
        else:
            # Convertir la imagen capturada a un array de OpenCV
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # Detectar la cara en la imagen
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=DETECTION_SCALE_FACTOR,
                                                  minNeighbors=DETECTION_MIN_NEIGHBORS,
                                                  minSize=DETECTION_MIN_SIZE)
            if len(faces) != 1:
                st.error(f"Para registrar, debe haber exactamente una cara (se detectaron: {len(faces)})")
            else:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                result = register_face_api(face_img, user_name)
                if "message" in result:
                    # Asigna un color si aún no se tiene para este nombre
                    if user_name not in st.session_state.registered_colors:
                        st.session_state.registered_colors[user_name] = generate_random_color()
                    st.success("Registro exitoso!")
                    # Guardamos el registro: fecha/hora, nombre, color y la imagen en base64
                    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _, encoded_img = cv2.imencode('.jpg', face_img)
                    face_img_b64 = base64.b64encode(encoded_img).decode("utf-8")
                    st.session_state.registrations.append({
                        "time": time_str,
                        "name": user_name,
                        "color": st.session_state.registered_colors[user_name],
                        "image_b64": face_img_b64
                    })
                else:
                    st.error("Error en el registro: " + str(result))
                    
#########################
# Vista 2: Identificación
#########################
elif view == "Identificación":
    st.title("Identificación en Tiempo Real")
    st.write("Activa o detén la cámara para identificar rostros en directo.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Activar Cámara"):
            st.session_state.identification_active = True
    with col2:
        if st.button("Detener Cámara"):
            st.session_state.identification_active = False

    placeholder = st.empty()       # Para mostrar el feed de video
    legend_placeholder = st.empty()  # Para mostrar la leyenda de etiquetas

    if st.session_state.identification_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No se pudo abrir la cámara.")
        else:
            last_identification = time.time()
            current_labels = []  # Lista que se mantiene a lo largo de los frames

            while st.session_state.identification_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo capturar el frame de la cámara.")
                    break

                # Efecto espejo
                frame = cv2.flip(frame, 1)

                # Detección de rostros en el frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=DETECTION_SCALE_FACTOR,
                    minNeighbors=DETECTION_MIN_NEIGHBORS,
                    minSize=DETECTION_MIN_SIZE
                )

                # Si current_labels está vacío o el número de rostros ha cambiado, lo inicializamos
                if not current_labels or len(current_labels) != len(faces):
                    current_labels = ["Desconocido"] * len(faces)

                # Si han pasado 5 segundos, actualizamos la identificación para todos los rostros
                if time.time() - last_identification >= 5:
                    nuevas_etiquetas = []
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        res = identify_face_api(face_img)
                        nuevas_etiquetas.append(res.get("name", "Desconocido"))
                    current_labels = nuevas_etiquetas
                    last_identification = time.time()

                # Dibujar los bounding boxes y las etiquetas según current_labels (se mantiene durante 5 segundos)
                for i, (x, y, w, h) in enumerate(faces):
                    label = current_labels[i]
                    if label in ["Desconocido", "Error"]:
                        box_color = UNKNOWN_COLOR
                    else:
                        # Asigna un color único si aún no se tiene para ese nombre
                        if label not in st.session_state.registered_colors:
                            st.session_state.registered_colors[label] = generate_random_color()
                        box_color = st.session_state.registered_colors[label]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

                # Actualizar el feed de video y la leyenda
                placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                legend_placeholder.text("Rostros detectados: " + ", ".join(current_labels))

                # Pausa corta para no sobrecargar la CPU
                time.sleep(0.03)

            cap.release()
    else:
        st.info("La cámara está detenida.")

#########################
# Vista 3: Historial
#########################
elif view == "Historial":
    st.title("Historial de Registros")
    if len(st.session_state.registrations) == 0:
        st.info("No hay registros aún.")
    else:
        for reg in st.session_state.registrations:
            st.write(f"**Fecha y Hora:** {reg['time']}")
            st.write(f"**Nombre:** {reg['name']}")
            st.write(f"**Color asignado (BGR):** {reg['color']}")
            st.image("data:image/jpeg;base64," + reg["image_b64"])
            st.markdown("---")
