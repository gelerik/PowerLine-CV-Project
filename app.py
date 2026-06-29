import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import subprocess
import time
import urllib.request

# --- DEVOPS_HACK: Запуск FastAPI внутри Streamlit Cloud ---
def start_backend():
    try:
        # Проверяем, отвечает ли FastAPI
        urllib.request.urlopen("http://127.0.0.1:8000/")
    except:
        # Если не отвечает, запускаем его скрытым процессом
        subprocess.Popen(["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
        time.sleep(3) # Ждем 3 секунды, чтобы сервер успел проснуться

start_backend()

# URL FastAPI сервера
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Детекция ЛЭП", page_icon="⚡", layout="wide")

st.title("⚡ Детекция повреждений ЛЭП")
st.write("Загрузите фотографию изоляторов для автоматического анализа дефектов.")

# --- БОКОВАЯ ПАНЕЛЬ (Выбор модели для требования №4) ---
st.sidebar.header("Настройки")
model_choice = st.sidebar.radio(
    "Выберите нейросеть:",
    ("Быстрая (YOLOv8n)", "Точная (YOLO26s)")
)

# Переводим выбор UI в понятный для API формат
model_type = "fast" if "Быстрая" in model_choice else "accurate"

# --- ОСНОВНАЯ ОБЛАСТЬ ---
# Загрузчик файлов
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Отображаем исходную картинку
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_column_width=True)

    # Кнопка для старта анализа
    if st.button("🔍 Анализировать", type="primary", use_container_width=True):
        with st.spinner('Сеть анализирует изображение...'):
            try:
                # Отправляем POST запрос на наш FastAPI
                files = {"file": uploaded_file.getvalue()}
                data = {"model_type": model_type}
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    detections = result["detections"]
                    
                    # Инструмент для рисования поверх картинки
                    draw = ImageDraw.Draw(image)
                    
                    # Задаем цвета для разных классов
                    colors = {
                        "insulator": "green",
                        "broken": "red",
                        "pollution-flashover": "orange"
                    }
                    
                    # Рисуем рамки
                    for det in detections:
                        box = det["box"] # [x1, y1, x2, y2]
                        class_name = det["class_name"]
                        conf = det["confidence"]
                        
                        color = colors.get(class_name, "red") # По умолчанию красный
                        
                        # Рисуем прямоугольник
                        draw.rectangle(box, outline=color, width=4)
                        
                        # Рисуем подпись
                        label = f"{class_name} {conf:.2f}"
                        # Рисуем черный фон для текста, чтобы было видно
                        draw.rectangle([box[0], box[1]-15, box[0]+100, box[1]], fill=color)
                        draw.text((box[0] + 2, box[1] - 15), label, fill="white")
                    
                    # Показываем результат
                    with col2:
                        st.subheader(f"Результат (Модель: {result['model_used']})")
                        st.image(image, use_column_width=True)
                        
                        if len(detections) == 0:
                            st.success("Дефектов не найдено (или объектов нет).")
                        else:
                            st.warning(f"Найдено объектов: {len(detections)}")
                            
                else:
                    st.error(f"Ошибка сервера: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Не удалось подключиться к Backend'у. Проверьте, что FastAPI сервер запущен!")