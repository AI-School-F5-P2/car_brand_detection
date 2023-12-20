import requests
from pathlib import Path
import streamlit as st
from PIL import Image
import subprocess
import os
from io import BytesIO

def get_subdirs(b='.'):
    '''
    Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
    Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def check_folders():
    paths = {
        'data_path': 'data',
        'images_path': 'data/images',
        'videos_path': 'data/videos'
    }
    # Check whether the specified path exists or not
    notExist = list(({file_type: path for (file_type, path) in paths.items() if not os.path.exists(path)}).values())

    if notExist:
        st.warning(f'Folders {notExist} do not exist. Creating now')
        # Create a new directory because it does not exist
        for folder in notExist:
            os.makedirs(folder)
            st.write(f"The new directory {folder} is created!")

# This will check if we have all the folders to save our files for inference
check_folders()

is_valid = False

if __name__ == '__main__':
    st.title('Car Brand Detection')

    source = ("Upload Image", "Image URL")
    source_index = st.sidebar.selectbox("Select Input type", range(len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("Load File", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading...'):
                st.sidebar.image(uploaded_file, caption='Imagen cargada')
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        url = st.text_input('Enter Image URL')
        if url:
            try:
                image = Image.open(BytesIO(requests.get(url).content))
                is_valid = True
                with st.spinner(text='Loading...'):
                    st.sidebar.image(image, caption='Imagen cargada')

                    # Guarda la imagen en una carpeta local para su posterior uso
                    image_filename = url.split("/")[-1]  # Obtiene el nombre de archivo de la URL
                    local_image_path = os.path.join("data", "images", image_filename)

                    # Guarda la imagen descargada localmente
                    image.save(local_image_path)

                    # Establece la ruta de la imagen como la fuente (source)
                    source = local_image_path
            except Exception as e:
                st.error(f"Ocurrió un error al cargar la imagen desde la URL: {e}")
                is_valid = False

    if is_valid:
        print('valid')
        if st.button('Detect'):
            with st.spinner(text='Running detection...'):
                process = subprocess.run(['yolo', 'task=detect', 'mode=predict', 'model=best.pt', 'conf=0.5', 'source={}'.format(source)],
                                     capture_output=True, universal_newlines=True)

            if process.stderr:
                st.error(process.stderr)
            else:
                st.success("Detection completed successfully!")

            with st.spinner(text='Preparing Image'):
                for file in os.listdir(get_detection_folder()):
                    # Mostrar imágenes desde la carpeta de detección
                    file_path = str(Path(get_detection_folder()) / file)
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        st.image(file_path)

                st.snow()
