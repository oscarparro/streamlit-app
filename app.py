import streamlit as st
import cv2
import numpy as np
import time
import requests
import random
import base64
import datetime
import face_recognition
from io import BytesIO
from PIL import Image

# Configuración de la página y estilos CSS (sin tocar)
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
API_URL = "http://127.0.0.1:8000"

prototxt_path = "model/deploy.prototxt"
caffe_model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

MODE_INTERVALS = {
    "Deteccion completa": 10,
    "Deteccion emociones": 2,
    "Deteccion rostros": 5,
    "No hacer nada": 0
}

UNKNOWN_COLOR = (0, 255, 0)


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
        fixed_color_str = "rgb(255, 0, 0)"
        for idx, (_, emotion) in enumerate(labels):
            html += f"- Emoción: {emotion} <span style='display:inline-block;width:12px;height:12px;background-color:{fixed_color_str};border:1px solid #000;margin-left:4px;vertical-align:middle;'></span><br>"
    else:
        html += "Modo no reconocido."
        
    return html

# ------------------------------------------------------
# CARGA DE REGISTROS PERSISTENTES
# ------------------------------------------------------
try:
    resp = requests.get(API_URL + "/list_faces")
    if resp.status_code == 200:
        st.session_state.registrations = resp.json()
    else:
        st.session_state.registrations = []
except:
    st.session_state.registrations = []

# Reconstruimos el mapa de colores
st.session_state.registered_colors = {
    rec["name"]: tuple(rec["color"])
    for rec in st.session_state.registrations
}

# Inicialización (solo la primera vez) de estos flags
if "identification_active" not in st.session_state:
    st.session_state.identification_active = False
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Deteccion completa"

#################
# TÍTULO Y TABS
#################
st.markdown('<div class="big-center-title">Proyecto de Detección e Identificación de Rostros</div>', unsafe_allow_html=True)
st.markdown("""
    <style>
    [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        font-size: 30px !important;
        font-weight: bold !important;
        padding: 0.5rem 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Identificación en Tiempo Real", "Registro de Rostro", "Historial de Registros"])

##########################################
# PESTAÑA 1: IDENTIFICACIÓN EN TIEMPO REAL
##########################################
with tab1:
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            height: 50px;
            font-size: 18px;
            display: block;
            margin: 0 auto;
        }
        .mode-selector {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- FORM para seleccionar y aplicar el modo ---
    with st.form("mode_form", clear_on_submit=False):
        cols = st.columns([3, 1], gap="small")
        with cols[0]:
            selected_mode = st.selectbox("",
                label_visibility="collapsed",
                options=list(MODE_INTERVALS.keys()),
                index=list(MODE_INTERVALS.keys()).index(st.session_state.current_mode),
                key="sel_mode",
            )
        with cols[1]:
            apply = st.form_submit_button("Aplicar Modo")

    if apply:
        # solo al hacer click lo actualizamos
        if selected_mode != st.session_state.current_mode:
            st.session_state.current_mode = selected_mode
            st.session_state.identification_active = False

    # --- Botones para activar / detener cámara ---
    col1, col2, col3 = st.columns([1,2,1], gap="small")
    with col2:
        subcolA, subcolB = st.columns([1,1], gap="small")
        with subcolA:
            if st.button("Activar Cámara", key="id_activate"):
                st.session_state.identification_active = True
        with subcolB:
            if st.button("Detener Cámara", key="id_deactivate"):
                st.session_state.identification_active = False

    # --- Placeholders para el feed y la leyenda ---
    st.markdown('<div class="camera-block">', unsafe_allow_html=True)
    feed_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    legend_placeholder = st.empty()

    # --- Lógica de captura y dibujado unificado ---
    if st.session_state.identification_active:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            st.error("No se pudo abrir la cámara.")
        else:
            last_ident = time.time()
            labels = []
            interval = MODE_INTERVALS[st.session_state.current_mode]

            while st.session_state.identification_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo capturar el frame.")
                    break

                frame = cv2.flip(frame, 1)
                faces = detect_faces_dnn(frame, net)

                if len(labels) != len(faces):
                    labels = [("Desconocido","N/A")] * len(faces)

                if interval > 0 and time.time() - last_ident >= interval:
                    new_labels = []
                    ml = st.session_state.current_mode.lower().strip()
                    for (x,y,w,h) in faces:
                        face = frame[y:y+h, x:x+w]
                        _, buf = cv2.imencode('.jpg', face)
                        res = requests.post(
                            API_URL+"/identify_face",
                            files={'file':('face.jpg',buf.tobytes(),'image/jpeg')},
                            data={'mode': st.session_state.current_mode}
                        )
                        try:
                            j = res.json()
                        except:
                            j = {"name":"Error","emotion":"N/A"}
                        new_labels.append((j.get("name","Desconocido"), j.get("emotion","N/A")))
                    labels = new_labels
                    last_ident = time.time()

                for i,(x,y,w,h) in enumerate(faces):
                    name, emo = labels[i]
                    ml = st.session_state.current_mode.lower().strip()
                    if "emociones" in ml:
                        color = (0,0,255); text=f"Emoción: {emo}"
                    else:
                        color = st.session_state.registered_colors.get(name, UNKNOWN_COLOR)
                        text = f"{name} ({emo})" if "completa" in ml else name
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame, text, (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                feed_placeholder.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), channels="RGB")
                legend_placeholder.markdown(format_detected_labels(labels, st.session_state.current_mode),
                                           unsafe_allow_html=True)
                time.sleep(0.03)
            cap.release()
    else:
        st.info("La cámara está detenida.")

#########################
# PESTAÑA 2: REGISTRO
#########################
with tab2:
    # 1) Inicializa el flag (solo la primera vez)
    if "register_mode" not in st.session_state:
        st.session_state.register_mode = False

    # 2) Botón que alterna entre mostrar/ocultar
    if st.button("Activar Registro", key="toggle_register"):
        st.session_state.register_mode = not st.session_state.register_mode

    # 3) Si el flag está activo, mostramos todo el UI de captura
    if st.session_state.register_mode:
        st.markdown('<div class="camera-block">', unsafe_allow_html=True)
        img_file = st.camera_input("Captura de Rostro", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
            .centered-button-container {
                display: flex;
                justify-content: center;
                margin-top: 1rem;
            }
            .centered-button-container button {
                width: 50% !important;
                height: 30px !important;
                font-size: 16px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        user_name = st.text_input("Nombre:", placeholder="Ingresa tu nombre", max_chars=20, key="reg_name")

        st.markdown('<div class="centered-button-container">', unsafe_allow_html=True)
        clicked = st.button("Registrar", key="reg_submit")
        st.markdown('</div>', unsafe_allow_html=True)

        if clicked:
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
                    x, y, w, h = faces[0]
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

                    if "record" in result:
                        rec = result["record"]
                        st.session_state.registrations.append(rec)
                        st.session_state.registered_colors[rec["name"]] = tuple(rec["color"])
                        st.success("¡Registro exitoso!")
                    else:
                        st.error("Error en el registro: " + str(result))

#########################
# PESTAÑA 3: HISTORIAL
#########################
with tab3:
    try:
        resp = requests.get(API_URL + "/list_faces")
        regs = resp.json() if resp.status_code == 200 else []
    except:
        st.error("Actualiza para cargar los rostros registrados.")
        regs = []

    if not regs:
        st.info("No hay registros aún.")
    else:
        for reg in regs:
            # Tres columnas: imagen, datos y botón
            col_img, col_data, col_btn = st.columns([1, 4, 1])

            # Imagen (decodificamos base64)
            with col_img:
                img_bytes = base64.b64decode(reg["image_b64"])
                st.image(img_bytes, width=100)

            # Datos
            with col_data:
                r, g, b = reg["color"]
                color_str = f"rgb({b},{g},{r})"
                st.markdown(f"""
                            **Fecha y Hora:** {reg['time']}  
                            **Nombre:** {reg['name']}  
                            **Color:** <span style='display:inline-block;width:12px;height:12px;background-color:{color_str};border:1px solid #000;vertical-align:middle;'></span>
                            """, unsafe_allow_html=True)

            # Botón pequeño integrado
            with col_btn:
                if st.button("Eliminar", key=reg["id"]):
                    d = requests.post(API_URL + "/delete_face", data={"id": reg["id"]})
                    if d.status_code == 200:
                        st.success(f"Registro eliminado.")
                    else:
                        st.error("Error al eliminar el registro.")

            # Línea separadora
            st.markdown("<hr>", unsafe_allow_html=True)