import streamlit as st
import cv2
import numpy as np
import time
import requests
import random
import base64
import datetime
import face_recognition

# Configuración de la página y estilos CSS
st.set_page_config(
    page_title="Face and Emotion Recognition System",
    page_icon="📷",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container { padding-top: 0.5rem !important; }
.big-center-title { text-align: center; font-size: 2.0rem; font-weight: bold; margin-top: 1.5rem; margin-bottom: 0.5em; }
.stTabs [data-baseweb="tab-list"] { justify-content: center; }
.stTabs [data-baseweb="tab-list"] button { font-size: 1.2rem; }
.camera-block { max-width: 400px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

#######################################################
# CONFIGURACIÓN GLOBAL Y VARIABLES
#######################################################
API_URL = "http://127.0.0.1:8000"  # Ajusta según sea necesario

prototxt_path = "model/deploy.prototxt"
caffe_model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# Intervalos de consulta a la API según modo
MODE_INTERVALS = {
    "Deteccion completa": 10,
    "Deteccion emociones": 2,
    "Deteccion rostros": 5,
    "No hacer nada": 0
}

UNKNOWN_COLOR = (0, 255, 0)  # Color predeterminado (verde)

# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------
def detect_faces_dnn(frame, net, conf_threshold=0.7):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def format_detected_labels(labels, mode):
    """
    Formatea la leyenda según el modo:
      - Deteccion completa: muestra nombre, emoción y color.
      - Deteccion rostros: muestra solo el nombre y el color.
      - Deteccion emociones: muestra la emoción y un color fijo (rojo) para todos los bounding boxes.
    """
    html = "<b>Resultados:</b><br><br>"
    mode_lower = mode.lower().strip()
    
    if mode_lower == "deteccion completa":
        for name, emotion in labels:
            if name not in ["Desconocido", "Error", "", "No detectado"]:
                if name in st.session_state.registered_colors:
                    b, g, r = st.session_state.registered_colors[name]
                    color_str = f"rgb({r}, {g}, {b})"
                    html += f"- {name} (Emoción: {emotion}) <span style='display:inline-block;width:12px;height:12px;background-color:{color_str};border:1px solid #000;margin-left:4px;vertical-align:middle;'></span><br>"
                else:
                    html += f"- {name} (Emoción: {emotion})<br>"
            else:
                html += f"- {name} (Emoción: {emotion})<br>"
                
    elif mode_lower == "deteccion rostros":
        for name, _ in labels:
            if name not in ["Desconocido", "Error", "", "No detectado"]:
                if name in st.session_state.registered_colors:
                    b, g, r = st.session_state.registered_colors[name]
                    color_str = f"rgb({r}, {g}, {b})"
                    html += f"- {name} <span style='display:inline-block;width:12px;height:12px;background-color:{color_str};border:1px solid #000;margin-left:4px;vertical-align:middle;'></span><br>"
                else:
                    html += f"- {name}<br>"
            else:
                html += f"- {name}<br>"

    elif mode_lower == "deteccion emociones":
        # Se usa color fijo (rojo) para todos los bounding boxes
        fixed_color_str = "rgb(255, 0, 0)"
        for idx, (_, emotion) in enumerate(labels):
            html += f"- Emoción: {emotion} <span style='display:inline-block;width:12px;height:12px;background-color:{fixed_color_str};border:1px solid #000;margin-left:4px;vertical-align:middle;'></span><br>"
    else:
        html += "Modo no reconocido."
        
    return html

# ------------------------------------------------------
# VARIABLES DE SESIÓN
# ------------------------------------------------------
if "registrations" not in st.session_state:
    st.session_state.registrations = []
if "registered_colors" not in st.session_state:
    st.session_state.registered_colors = {}
if "identification_active" not in st.session_state:
    st.session_state.identification_active = False
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Deteccion completa"

#######################################################
# TÍTULO
#######################################################
st.markdown('<div class="big-center-title">Proyecto de Detección e Identificación de Rostros</div>', unsafe_allow_html=True)

#######################################################
# CREAR TABS
#######################################################
tab1, tab2, tab3 = st.tabs(["Registro de Rostro", "Identificación en Tiempo Real", "Historial"])

#########################
# PESTAÑA 1: REGISTRO
#########################
with tab1:
    st.markdown('<div class="camera-block">', unsafe_allow_html=True)
    img_file = st.camera_input("Captura de Rostro", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    user_name = st.text_input("Nombre:", placeholder="Ingresa tu nombre", max_chars=20)
    if st.button("Registrar"):
        if img_file is None:
            st.error("Por favor, toma una foto primero.")
        elif user_name.strip() == "":
            st.error("Ingresa un nombre.")
        else:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            faces = detect_faces_dnn(frame, net, conf_threshold=0.7)
            if len(faces) != 1:
                st.error(f"Para registrar, debe haber exactamente una cara (detectadas: {len(faces)})")
            else:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                _, img_encoded = cv2.imencode('.jpg', face_img)
                files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
                data = {'name': user_name}
                res = requests.post(API_URL + "/register_face", files=files, data=data)
                try:
                    result = res.json()
                except Exception:
                    st.error("Error al decodificar la respuesta del servidor: " + res.text)
                    st.stop()
                if "message" in result:
                    if user_name not in st.session_state.registered_colors:
                        st.session_state.registered_colors[user_name] = generate_random_color()
                    st.success("¡Registro exitoso!")
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
# PESTAÑA 2: IDENTIFICACIÓN EN TIEMPO REAL
#########################
with tab2:
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            height: 50px;
            font-size: 18px;
            margin: 5px auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Selección de modo
    modo = st.selectbox(
        "Selecciona el modo de funcionamiento",
        options=["Deteccion completa", "Deteccion emociones", "Deteccion rostros", "No hacer nada"],
        index=["Deteccion completa", "Deteccion emociones", "Deteccion rostros", "No hacer nada"].index(st.session_state.current_mode)
    )
    # Si el modo cambia, se reinicia la cámara para aplicar la nueva configuración
    if modo != st.session_state.current_mode:
        st.session_state.current_mode = modo
        st.session_state.identification_active = False
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    with col2:
        if st.button("Activar Cámara"):
            st.session_state.identification_active = True
        if st.button("Detener Cámara"):
            st.session_state.identification_active = False

    st.markdown('<div class="camera-block">', unsafe_allow_html=True)
    feed_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    legend_placeholder = st.empty()

    if st.session_state.identification_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No se pudo abrir la cámara.")
        else:
            last_identification = time.time()
            current_labels = []
            interval = MODE_INTERVALS[st.session_state.current_mode]

            while st.session_state.identification_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo capturar el frame.")
                    break

                frame = cv2.flip(frame, 1)  # Vista espejo
                faces = detect_faces_dnn(frame, net, conf_threshold=0.7)

                # Actualiza la lista de labels si cambia el número de rostros
                if len(current_labels) != len(faces):
                    current_labels = [("Desconocido", "N/A")] * len(faces)

                # Consulta a la API si ha transcurrido el intervalo definido
                if interval > 0 and time.time() - last_identification >= interval:
                    nuevas_etiquetas = []
                    mode_lower = st.session_state.current_mode.lower().strip()
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        _, img_encoded = cv2.imencode('.jpg', face_img)
                        files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
                        data = {'mode': st.session_state.current_mode}
                        res = requests.post(API_URL + "/identify_face", files=files, data=data)
                        try:
                            res_json = res.json()
                        except Exception:
                            res_json = {"name": "Error", "emotion": "N/A"}
                        
                        if mode_lower == "deteccion completa":
                            name = res_json.get("name", "Desconocido")
                            emotion = res_json.get("emotion", "N/A")
                            nuevas_etiquetas.append((name, emotion))
                        elif mode_lower == "deteccion emociones":
                            name = res_json.get("name", "N/A")
                            emotion = res_json.get("emotion", "N/A")
                            nuevas_etiquetas.append((name, emotion))
                        elif mode_lower == "deteccion rostros":
                            if "error" in res_json:
                                nuevas_etiquetas.append(("No detectado", ""))
                            else:
                                name = res_json.get("name", "Desconocido")
                                nuevas_etiquetas.append((name, ""))
                        elif mode_lower == "no hacer nada":
                            nuevas_etiquetas.append(("", ""))
                    
                    current_labels = nuevas_etiquetas
                    last_identification = time.time()

                # Dibujar bounding boxes y mostrar leyenda
                for i, (x, y, w, h) in enumerate(faces):
                    label, emotion = current_labels[i]
                    mode_lower = st.session_state.current_mode.lower().strip()
                    if mode_lower == "deteccion emociones":
                        # Fijamos el color rojo (en BGR: (0, 0, 255)) para todos los bounding boxes
                        box_color = (0, 0, 255)
                        text = f"Emocion: {emotion}"
                    elif mode_lower == "deteccion rostros":
                        if label not in ["Desconocido", "Error", "", "No detectado"]:
                            if label not in st.session_state.registered_colors:
                                st.session_state.registered_colors[label] = generate_random_color()
                            box_color = st.session_state.registered_colors[label]
                        else:
                            box_color = UNKNOWN_COLOR
                        text = f"{label}"
                    else:  # Deteccion completa
                        if label not in ["Desconocido", "Error", "", "No detectado"]:
                            if label not in st.session_state.registered_colors:
                                st.session_state.registered_colors[label] = generate_random_color()
                            box_color = st.session_state.registered_colors[label]
                        else:
                            box_color = UNKNOWN_COLOR
                        text = f"{label} ({emotion})"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(frame, text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

                feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                legend_placeholder.markdown(format_detected_labels(current_labels, st.session_state.current_mode), unsafe_allow_html=True)
                time.sleep(0.03)
            cap.release()
    else:
        st.info("La cámara está detenida.")

#########################
# PESTAÑA 3: HISTORIAL
#########################
with tab3:
    if len(st.session_state.registrations) == 0:
        st.info("No hay registros aún.")
    else:
        for reg in st.session_state.registrations:
            b, g, r = reg["color"]
            color_str = f"rgb({r},{g},{b})"
            html = f"""
            <div style="display: flex; align-items: center; margin-bottom: 1em; border-bottom: 1px solid #ddd; padding-bottom: 1em;">
                <div style="flex: 1;">
                    <img src="data:image/jpeg;base64,{reg['image_b64']}" style="width:200px; height:auto;" />
                </div>
                <div style="flex: 2; margin-left: 1em; font-size: 1rem;">
                    <p style="margin: 0;"><b>Fecha y Hora:</b> {reg['time']}</p>
                    <p style="margin: 0;"><b>Nombre:</b> {reg['name']}</p>
                    <p style="margin: 0;">
                        <b>Color:</b> 
                        <span style="display:inline-block;width:12px;height:12px;background-color:{color_str};border:1px solid #000;margin-left:4px;vertical-align:middle;"></span>
                    </p>
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
