import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os

# Настройки страницы Streamlit
st.set_page_config(page_title="Детекция ЛЭП", page_icon="⚡", layout="wide")

# ПУТИ К МОДЕЛЯМ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAST_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n(fast).pt")
ACCURATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo26s(accurate).pt")

# КЭШИРОВАНИЕ МОДЕЛЕЙ (СПАСАЕТ ОПЕРАТИВНУЮ ПАМЯТЬ)
@st.cache_resource
def load_model(model_type):
    if model_type == "fast":
        return YOLO(FAST_MODEL_PATH)
    else:
        return YOLO(ACCURATE_MODEL_PATH)

st.title("⚡ Детекция повреждений ЛЭП")
st.write("Загрузите фотографию изоляторов для автоматического анализа дефектов.")

# БОКОВАЯ ПАНЕЛЬ
st.sidebar.header("Настройки")
model_choice = st.sidebar.radio(
    "Выберите нейросеть:",
    ("Быстрая (YOLOv8n)", "Точная (YOLOv8m)")
)
model_type = "fast" if "Быстрая" in model_choice else "accurate"

# ОСНОВНАЯ ОБЛАСТЬ
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_column_width=True)

    if st.button("🔍 Анализировать", type="primary", use_container_width=True):
        with st.spinner(f'Загрузка {model_choice} и анализ...'):
            try:
                # Загружаем нужную модель напрямую (без FastAPI)
                model = load_model(model_type)
                
                # Делаем предсказание
                results = model.predict(image, conf=0.25)
                
                # Рисуем рамки
                draw = ImageDraw.Draw(image)
                colors = {"insulator": "green", "broken": "red", "pollution-flashover": "orange"}
                
                detections_count = 0
                
                for r in results:
                    for box in r.boxes:
                        detections_count += 1
                        class_name = model.names[int(box.cls)]
                        conf = float(box.conf)
                        coords = [float(x) for x in box.xyxy[0]]
                        
                        color = colors.get(class_name, "red")
                        draw.rectangle(coords, outline=color, width=4)
                        
                        label = f"{class_name} {conf:.2f}"
                        draw.rectangle([coords[0], coords[1]-15, coords[0]+100, coords[1]], fill=color)
                        draw.text((coords[0] + 2, coords[1] - 15), label, fill="white")
                
                # Показываем результат
                with col2:
                    st.subheader("Результат")
                    st.image(image, use_column_width=True)
                    
                    if detections_count == 0:
                        st.success("Дефектов не найдено (или объектов нет).")
                    else:
                        st.warning(f"Найдено объектов: {detections_count}")
                        
            except Exception as e:
                st.error(f"Произошла ошибка при анализе: {str(e)}")