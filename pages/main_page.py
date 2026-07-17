import streamlit as st
import pandas as pd

# Минимальные стили через встроенный markdown (без сторонних библиотек)
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .model-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #dee2e6;
        height: 100%;
    }
    .model-card h3 { margin-top: 0; color: #1a1a2e; }
    .hero-placeholder {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
        text-align: center;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# ДАННЫЕ ПРОЕКТА
# =============================================================================

# Заголовок и описание проекта
PROJECT_TITLE = "Обнаружение дефектов ЛЭП с помощью YOLO"
PROJECT_DESCRIPTION = [
    "Индивидуальная проектная работа по автоматическому обнаружению дефектов на линиях электропередач (ЛЭП) с использованием моделей компьтерного зрения.",
]

# Карточки моделей
MODELS = [
    {
        "name": "YOLOv12n",
        "description": (
            "Компактная модель для быстрого инференса. "
            "Оптимальна для edge-устройств и потоковой обработки видео."
        ),
        "badge": "Small",
    },
    {
        "name": "YOLOv12s",
        "description": (
            "Оптимальный баланс между точностью и скоростью инференса. "
            "Подходит для сценариев с ограниченными ресурсами GPU."
        ),
        "badge": "Medium",
    },
    {
        "name": "YOLOv12L",
        "description": (
            "Тяжелая и точная модель для детального офлайн-анализа "
            "и задач, где критична полнота обнаружения дефектов."
        ),
        "badge": "Large",
    },
]


# Таблица сравнения моделей
MODEL_COMPARISON = pd.DataFrame(
    {
        "Название модели": ["YOLOv12n", "YOLOv12s", "YOLOv12L"],
        "mAP50": [0.834, 0.854, 0.883],
        "mAP50-95": [0.573, 0.591, 0.681],
        "Размер модели (кол-во параметров, размер .pt)": ["2.6 млн, 6 МБ", "9.3 млн, 20 МБ", "26.45 млн, 54 МБ"],
        "Скорость инференса (Nvidia T4)": ["~28 мс", "~69 мс", "~141 мс"],
    }
)

# Классы обнаруживаемых объектов
HEALTHY_CLASSES = [
    ("🟢", "vibration_damper", "Демпфер вибрации"),
    ("🟢", "festoon_insulator", "Фестонный изолятор"),
    ("🟢", "polymer_insulator", "Полимерный изолятор"),
    ("🟢", "traverse", "Перекладина"),
    ("🟢", "safety_sign", "Знак безопасности"),
]

DEFECT_CLASSES = [
    ("🔴", "bad_insulator", "Плохой изолятор"),
    ("🔴", "damaged_insulator", "Повреждённый изолятор"),
    ("🟠", "nest", "Гнездо птиц"),
]

# Метрики по классам (отдельная таблица для каждой модели)
_CLASS_NAMES = [
    "bad_insulator",
    "damaged_insulator",
    "nest",
    "safety_sign+",
    "festoon_insulators",
    "traverse",
    "vibration_damper",
    "polymer_insulators"
]

CLASS_METRICS_BY_MODEL = {
    "YOLOv12L": pd.DataFrame(
        {
            "Класс": _CLASS_NAMES,
            "mAP50": [0.979, 0.966, 0.855, 0.914, 0.907, 0.892, 0.812, 0.742],
            "mAP50-95": [0.871, 0.864, 0.666, 0.567, 0.724, 0.705, 0.606, 0.445],
        }
    ),
    "YOLOv12s": pd.DataFrame(
        {
            "Класс": _CLASS_NAMES,
            "mAP50": [0.980, 0.930, 0.733, 0.915, 0.878, 0.863, 0.787, 0.742],
            "mAP50-95": [0.819, 0.782, 0.445, 0.557, 0.592, 0.620, 0.544, 0.365],
        }
    ),
    "YOLOv12n": pd.DataFrame(
        {
            "Класс": _CLASS_NAMES,
            "mAP50": [0.972, 0.908, 0.857, 0.854, 0.842, 0.769, 0.713, 0.642],
            "mAP50-95": [0.784, 0.763, 0.495, 0.584, 0.581, 0.521, 0.394, 0.303],
        }
    ),
}


# Параметры обучения (отдельно для каждой модели)
TRAINING_PARAMS_BY_MODEL = {
    "YOLOv12L": {
        "left": {
            "Количество изображений": "6390",
            "Размер изображения": "1024p",
            "Optimizer": "MuSGD",
            "Learning Rate": "0.01",
        },
        "right": {
            "GPU": "Nvidia A100",
            "Framework": "PyTorch",
            "Ultralytics Version": "8.4.86",
            "PyTorch Version": "3.12.13",
        },
    },
    "YOLOv12s": {
        "left": {
            "Количество изображений": "6390",
            "Размер изображения": "1280p",
            "Optimizer": "AdamW",
            "Learning Rate": "0.0005",
        },
        "right": {
            "GPU": "Nvidia T4 x2",
            "Framework": "PyTorch",
            "Ultralytics Version": "8.4.86",
            "PyTorch Version": "3.12.13",
        },
    },
    "YOLOv12n": {
        "left": {
            "Количество изображений": "6390",
            "Размер изображения": "1024p",
            "Optimizer": "AdamW",
            "Learning Rate": "0.001",
        },
        "right": {
            "GPU": "Nvidia T4 x2",
            "Framework": "PyTorch",
            "Ultralytics Version": "8.4.86",
            "PyTorch Version": "3.12.13",
        },
    },
}

# Общий список моделей для выпадающих списков
MODEL_OPTIONS = list(CLASS_METRICS_BY_MODEL.keys())


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def render_section_header(title: str) -> None:
    """Отрисовка заголовка секции с разделительной линией."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_model_card(model: dict) -> None:
    st.markdown(
        f"""
        <div class="model-card">
            <h3>{model["name"]}</h3>
            <span style="
                background:#667eea; color:white; padding:2px 10px;
                border-radius:12px; font-size:0.75rem;
            ">{model["badge"]}</span>
            <p style="margin-top:1rem; color:#495057; line-height:1.6;">
                {model["description"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_list(classes: list[tuple[str, str, str]]) -> None:
    for emoji, class_name, label in classes:
        st.markdown(f"{emoji} **`{class_name}`** — {label}")


def render_params_column(params: dict) -> None:
    for key, value in params.items():
        st.metric(label=key, value=value)


# СЕКЦИЯ 1: ЗАГОЛОВОК ПРОЕКТА
st.markdown("# ⚡ " + PROJECT_TITLE)

col_text, col_image = st.columns([3, 2])

with col_text:
    for paragraph in PROJECT_DESCRIPTION:
        st.markdown(paragraph)


# СЕКЦИЯ 2: ИСПОЛЬЗУЕМЫЕ МОДЕЛИ
render_section_header("Используемые модели")

model_cols = st.columns(len(MODELS))
for col, model in zip(model_cols, MODELS):
    with col:
        render_model_card(model)


# СЕКЦИЯ 3: МЕТРИКИ МОДЕЛЕЙ
st.subheader("Сравнение моделей")


st.dataframe(
    MODEL_COMPARISON,
    use_container_width=True,
    hide_index=True,
)



# СЕКЦИЯ 4: КЛАССЫ ОБНАРУЖИВАЕМЫХ ОБЪЕКТОВ
render_section_header("Классы обнаруживаемых объектов")

class_col1, class_col2 = st.columns(2)

with class_col1:
    st.subheader("Исправные элементы")
    render_class_list(HEALTHY_CLASSES)

with class_col2:
    st.subheader("Дефекты")
    render_class_list(DEFECT_CLASSES)

st.subheader("Метрики по классам")

selected_model = st.selectbox(
    "Модель",
    options=MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("YOLOv12s"),
    key="class_metrics_model",
)


st.dataframe(
    CLASS_METRICS_BY_MODEL[selected_model],
    use_container_width=True,
    hide_index=True,
)


# СЕКЦИЯ 5: ИНФОРМАЦИЯ ОБ ОБУЧЕНИИ
render_section_header("Информация об обучении")

selected_training_model = st.selectbox(
    "Модель",
    options=MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("YOLOv12s"),
    key="training_model",
)

train_col_left, train_col_right = st.columns(2)

with train_col_left:
    st.markdown("##### Гиперпараметры и датасет")
    render_params_column(TRAINING_PARAMS_BY_MODEL[selected_training_model]["left"])

with train_col_right:
    st.markdown("##### Инфраструктура и окружение")
    render_params_column(TRAINING_PARAMS_BY_MODEL[selected_training_model]["right"])



st.divider()
st.caption("Power Line Defect Detection · Potapov Egor · 2026")
