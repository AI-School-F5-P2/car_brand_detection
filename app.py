import requests
import streamlit as st
from PIL import Image
from io import BytesIO
import tempfile
from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8n model
model = YOLO('best.pt')
names = model.model.names

st.title('Predicción de YOLOv8 desde URL')

# Entrada para la URL de la imagen
url = st.text_input('Ingresa la URL de la imagen')
if url:
    try:

        # Realiza la predicción con el modelo
        results = model(url, show=True, conf=0.5, save=True)

        # Procesa las detecciones
        boxes = results[0].boxes.xywh.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        # Lee la imagen desde la URL
        image = Image.open(BytesIO(requests.get(url).content))
        st.image(image, caption='Imagen cargada', use_column_width=True)

        # Muestra los resultados de la predicción
        st.write('Resultados de la predicción:')
        for box, cls, conf in zip(boxes, clss, confs):
            st.write(f"Clase: {names[int(cls)]}, Confianza: {conf}, Caja: {box}")
    except Exception as e:
        st.error(f'Ocurrió un error al cargar o procesar la imagen: {e}')




