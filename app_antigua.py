import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import face_recognition
import pandas as pd
import time

# Módulos personalizados
from face_detection import FaceDetector
from face_embeddings import get_face_embedding
# Importamos la clase FaceRegistrar que acabamos de crear
from face_registration import FaceRegistrar

# -------------------------------------------------------------------
# 1. Inicialización en session_state
# -------------------------------------------------------------------
if "registered_faces_df" not in st.session_state:
    st.session_state["registered_faces_df"] = pd.DataFrame(columns=["Name", "Embedding"])
if "registration_frame" not in st.session_state:
    st.session_state["registration_frame"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "intro"  # "intro", "register", "main"
if "capture_request" not in st.session_state:
    st.session_state["capture_request"] = False

# -------------------------------------------------------------------
# 2. Configuración general de la página
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Face and Emotion Recognition System",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; height: 10vh;">
        <h1>Face and Emotion Recognition System</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# 3. Configuración del modelo de detección facial
# -------------------------------------------------------------------
FACE_MODEL_PROTO = "deploy.prototxt"
FACE_MODEL_CAFFE = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetector(FACE_MODEL_PROTO, FACE_MODEL_CAFFE, confidence_threshold=0.5)

# Creamos una instancia global del FaceRegistrar
face_registrar = FaceRegistrar(face_detector)

# -------------------------------------------------------------------
# 4. Página de introducción
# -------------------------------------------------------------------
def show_intro_page():
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 30vh;">
            <div style="text-align: left;">
                <ul>
                    <li>Esta aplicación detecta rostros en tiempo real.</li>
                    <li>Primero, regístrate para que podamos reconocer tu rostro.</li>
                    <li>Si ya te has registrado, se mostrará la cámara en vivo con reconocimiento.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .stButton button {
            width: 50%;
            height: 50px;
            font-size: 18px;
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    with col2:
        subcolA, subcolB = st.columns([1, 1], gap="small")
        with subcolA:
            if st.button("Aceptar"):
                st.session_state["page"] = "register"
        with subcolB:
            if st.button("Rechazar"):
                st.error("Has rechazado la aplicación. Redirigiendo...")
                st.markdown(
                    """<meta http-equiv="refresh" content="0; url=https://www.google.com" />""",
                    unsafe_allow_html=True,
                )
                st.stop()

# -------------------------------------------------------------------
# 5. Página de registro de rostro
# -------------------------------------------------------------------
def show_registration_page():
    st.subheader("Registro de Nuevo Rostro")

    class RegistrationTransformer(VideoProcessorBase):
        def __init__(self):
            self.face_detector = face_detector

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Dibujamos bounding boxes para feedback visual
            img_with_boxes = self.face_detector.detect_faces(img.copy())

            # Si se solicitó captura, guardamos el frame
            if st.session_state["capture_request"]:
                st.session_state["registration_frame"] = img.copy()
                st.session_state["capture_request"] = False

            return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")

    webrtc_streamer(
        key="register",
        video_processor_factory=RegistrationTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_html_attrs={
            "style": {"width": "60%", "height": "auto", "display": "block", "margin": "0 auto"},
            "autoPlay": True,
            "controls": False,
        },
    )

    # Botón para capturar un frame
    if st.button("Capturar"):
        st.session_state["capture_request"] = True

    captured_frame = st.session_state["registration_frame"]
    if captured_frame is not None:
        st.image(captured_frame, channels="BGR", caption="Frame Capturado")

        name_input = st.text_input("Ingresa tu nombre para registrar este rostro:")
        if st.button("Registrar"):
            # Llamamos a la lógica de FaceRegistrar
            result_message = face_registrar.register_face(captured_frame, name_input)
            if "registrada" in result_message.lower():
                st.success(result_message)
                # Cambiamos a la página principal
                st.session_state["page"] = "main"
                st.experimental_rerun()
            else:
                st.error(result_message)
    else:
        st.info("No se ha capturado ningún frame todavía. Pulsa **Capturar** para congelar un frame.")

# -------------------------------------------------------------------
# 6. Página principal (Reconocimiento en vivo)
# -------------------------------------------------------------------
def show_main_page():
    st.subheader("Reconocimiento Facial en Vivo")

    if st.button("Nuevo Registro"):
        st.session_state["page"] = "register"
        st.experimental_rerun()

    class RecognitionTransformer(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Detección
            img_with_boxes = face_detector.detect_faces(img.copy())
            boxes = face_detector.get_face_boxes(img)
            recognized_faces = []

            for (startX, startY, endX, endY) in boxes:
                face_roi = img[startY:endY, startX:endX]
                embedding = get_face_embedding(face_roi)
                name = "Desconocido"
                df = st.session_state["registered_faces_df"]
                if embedding is not None and not df.empty:
                    for idx, row in df.iterrows():
                        reg_embedding = row["Embedding"]
                        matches = face_recognition.compare_faces([reg_embedding], embedding, tolerance=0.6)
                        if matches[0]:
                            name = row["Name"]
                            break
                recognized_faces.append((name, (startX, startY, endX, endY)))

            for (name, (startX, startY, endX, endY)) in recognized_faces:
                cv2.putText(img_with_boxes, name, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")

    webrtc_streamer(
        key="main",
        video_processor_factory=RecognitionTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "audio": False,
        },
        video_html_attrs={
            "style": {"width": "60%", "height": "auto", "display": "block", "margin": "0 auto"},
            "autoPlay": True,
            "controls": False,
        },
    )

# -------------------------------------------------------------------
# 7. Lógica de navegación
# -------------------------------------------------------------------
if st.session_state["page"] == "intro":
    show_intro_page()
    st.stop()
elif st.session_state["page"] == "register":
    show_registration_page()
    st.stop()
elif st.session_state["page"] == "main":
    show_main_page()
