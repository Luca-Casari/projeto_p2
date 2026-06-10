import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import date
import os
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass, shift

# -----------------------
# Configuração da página
# -----------------------
st.set_page_config(
    page_title="SmartDigit KNN",
    page_icon="✍️",
    layout="wide"
)

# -----------------------
# CSS
# -----------------------

st.markdown("""
<style>

.stApp{
    background-color:#02111f;
}

.titulo{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:white;
}

.subtitulo{
    text-align:center;
    color:#2ee6b8;
    margin-bottom:30px;
}

.card{
    background:#071b33;
    border:1px solid #12385d;
    border-radius:10px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Carrega modelo
# -----------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "modelo" / "knn_model.pkl")

pca = joblib.load(BASE_DIR / "modelo" / "pca.pkl")

# -----------------------
# Cabeçalho
# -----------------------

st.markdown(
    "<div class='titulo'>SmartDigit KNN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitulo'>Classificador de Dígitos Manuscritos utilizando K-Nearest Neighbors</div>",
    unsafe_allow_html=True
)

# -----------------------
# Cards superiores
# -----------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
        <h5>Modelo</h5>
        <h2>KNN</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <h5>Arquivo</h5>
        <h2>knn_model.pkl</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <h5>Classes</h5>
        <h2>10</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
        <h5>Entrada</h5>
        <h2>28 x 28</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------
# Área principal
# -----------------------

col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("Imagem")

    canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:

    if st.button("Classificar"):

        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))

        img = img.convert("L")

        img_array = np.array(img)

        img_array = np.where(img_array > 80, 255, 0).astype(np.uint8)

        coords = np.argwhere(img_array > 0)

        if coords.size == 0:
            st.warning("Desenhe um dígito antes de classificar.")
            st.stop()

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img_array = img_array[
         y_min:y_max + 1,
         x_min:x_max + 1
        ]

        h, w = img_array.shape

        if h > w:

            novo_h = 20
            novo_w = max(1, round(w * 20 / h))

        else:

            novo_w = 20
            novo_h = max(1, round(h * 20 / w))

        img = Image.fromarray(img_array)

        img = img.resize(
            (novo_w, novo_h),
            Image.Resampling.LANCZOS
        )

        img_array = np.array(img)

        canvas = np.zeros((28, 28), dtype=np.uint8)

        y_offset = (28 - novo_h) // 2
        x_offset = (28 - novo_w) // 2

        canvas[
            y_offset:y_offset + novo_h,
            x_offset:x_offset + novo_w
        ] = img_array

        cy, cx = center_of_mass(canvas)

        desloc_x = 14 - cx
        desloc_y = 14 - cy

        canvas = shift(
            canvas,
            shift=(desloc_y, desloc_x),
            mode="constant",
            cval=0
        )

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        img = Image.fromarray(canvas)

        # Apenas para visualização
        st.image(
            img.resize(
                (280, 280),
                Image.Resampling.NEAREST
            ),
            caption="Imagem enviada ao modelo"
        )

        transform = transforms.ToTensor()
        normalize = transforms.Normalize((0.5,), (0.5,))

        tensor = transform(img)
        tensor = normalize(tensor)

        vetor = tensor.squeeze().numpy().flatten()

        vetor_pca = pca.transform([vetor])

        pred = model.predict(vetor_pca)[0]

        probs = model.predict_proba(vetor_pca)[0]
        confianca = probs[pred] * 100

        st.success(f"Dígito previsto: {pred}")

        st.metric(
            "Confiança",
            f"{confianca:.2f}%"
        )

with col2:

    st.subheader("Informações")

    st.write("**Dimensão:** 28 × 28 pixels")
    st.write("**Quantidade de atributos:** 50")
    st.write("**Modelo carregado:** knn_model.pkl")