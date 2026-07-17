"""
Страница «Детекция изображения» — Streamlit-приложение
обнаружения дефектов ЛЭП с помощью YOLO.

Запуск всего приложения: streamlit run app/main_page.py
"""

import io
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from history_manager import save_detection

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parents[1] / ".ultralytics"),
)

from ultralytics import YOLO

# =============================================================================
# КОНСТАНТЫ И ДАННЫЕ
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "YOLOv12n": ROOT_DIR / "models" / "yolo12n.pt",
    "YOLOv12s": ROOT_DIR / "models" / "yolo12s.pt",
    "YOLOv12L": ROOT_DIR / "models" / "yolo12L.pt",
}

MODEL_DESCRIPTIONS = {
    "YOLOv12n": (
        "Компактная модель YOLOv12n — минимальный размер и высокая скорость. "
        "Подходит для быстрого предварительного анализа на CPU."
    ),
    "YOLOv12s": (
        "YOLOv12s — баланс точности и скорости. Рекомендуется для повседневного "
        "использования на GPU среднего класса."
    ),
    "YOLOv12L": (
        "YOLOv12L — тяжелая и точная модель. Оптимальна для детального "
        "офлайн-анализа снимков ЛЭП."
    ),
}

# Порядок классов в таблице (фиксированный)
DANGEROUS_CLASSES = ["damaged_insulator", "bad_insulator", "nest"]
SAFE_CLASSES = [
    "polymer_insulator",
    "vibration_damper",
    "festoon_insulator",
    "traverse",
    "safety_sign",
]
ALL_CLASSES = DANGEROUS_CLASSES + SAFE_CLASSES

CLASS_LABELS_RU = {
    "damaged_insulator": "Повреждённый изолятор",
    "bad_insulator": "Плохой изолятор",
    "nest": "Гнездо",
    "polymer_insulator": "Полимерный изолятор",
    "vibration_damper": "Демпфер вибрации",
    "festoon_insulator": "Фестонный изолятор",
    "traverse": "Траверса",
    "safety_sign": "Знак безопасности",
}

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stSidebar"] .sidebar-info {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def build_detection_table(counts: dict[str, int]) -> pd.DataFrame:
    """Формирует таблицу обнаруженных объектов с фиксированным порядком классов."""
    rows = []
    ordered_classes = ALL_CLASSES + sorted(set(counts) - set(ALL_CLASSES))
    for class_name in ordered_classes:
        status = "⚠️ Дефект" if class_name in DANGEROUS_CLASSES else "✅ Норма"
        rows.append(
            {
                "Класс": class_name,
                "Название": CLASS_LABELS_RU.get(class_name, class_name),
                "Количество": counts.get(class_name, 0),
                "Статус": status,
            }
        )
    return pd.DataFrame(rows)


def style_detection_table(df: pd.DataFrame) -> Any:
    """Подсветка строк: опасные — светло-красный, безопасные — светло-зелёный."""
    dangerous_set = set(DANGEROUS_CLASSES)

    def highlight_row(row: pd.Series) -> list[str]:
        if row["Класс"] in dangerous_set:
            return ["background-color: #fde8e8; color: #111827"] * len(row)
        return ["background-color: #e8f5e9; color: #111827"] * len(row)

    return df.style.apply(highlight_row, axis=1).set_properties(
        **{
            "color": "#111827",
            "border-color": "#d1d5db",
        }
    )


def build_summary_text(counts: dict[str, int]) -> str:
    """Формирует текстовую сводку — только классы с count > 0."""
    lines = ["**Обнаружено:**", ""]
    ordered_classes = ALL_CLASSES + sorted(set(counts) - set(ALL_CLASSES))
    for class_name in ordered_classes:
        count = counts.get(class_name, 0)
        if count <= 0:
            continue
        emoji = "🔴" if class_name in DANGEROUS_CLASSES else "🟢"
        label = CLASS_LABELS_RU.get(class_name, class_name)
        lines.append(f"{emoji} **{class_name}** ({label}) — {count}")
    if len(lines) == 2:
        return "**Обнаружено:** объекты не найдены."
    return "\n\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> YOLO:
    """Загружает YOLO-модель один раз на выбранный путь весов."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    return YOLO(str(path))


def get_box_color(class_name: str) -> str:
    """Возвращает цвет рамки для класса."""
    return "#e53935" if class_name in DANGEROUS_CLASSES else "#43a047"


def draw_detection_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    color: str,
) -> None:
    """Рисует читаемый label над bbox."""
    if not label:
        return

    text_box = draw.textbbox((0, 0), label)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x1, y1, _, _ = box
    label_y1 = max(0, y1 - text_h - 8)
    label_box = (x1, label_y1, x1 + text_w + 8, label_y1 + text_h + 6)

    draw.rectangle(label_box, fill=color)
    draw.text((label_box[0] + 4, label_box[1] + 3), label, fill="#ffffff")


def run_detection(
    image: Image.Image,
    model_name: str,
    conf: float,
    iou: float,
    show_confidence: bool,
    show_class_names: bool,
) -> tuple[Image.Image, dict[str, int], float, list[dict[str, Any]]]:
    """Выполняет YOLO-инференс и рисует реальные bbox на изображении."""
    model = load_yolo_model(MODEL_PATHS[model_name])
    result_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result_image)
    line_width = max(2, round(min(result_image.size) / 250))

    start_time = perf_counter()
    results = model.predict(
        source=result_image,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    inference_ms = (perf_counter() - start_time) * 1000

    counts: Counter[str] = Counter()
    detected_objects = []
    names = getattr(model, "names", {})

    for result in results:
        result_names = getattr(result, "names", names)
        for detected_box in result.boxes:
            class_id = int(detected_box.cls.item())
            class_name = result_names.get(class_id, str(class_id))
            score = float(detected_box.conf.item())
            x1, y1, x2, y2 = detected_box.xyxy[0].tolist()
            box = (round(x1), round(y1), round(x2), round(y2))

            counts[class_name] += 1
            detected_objects.append(
                {
                    "class": class_name,
                    "confidence": score,
                    "bbox": box,
                }
            )
            color = get_box_color(class_name)
            draw.rectangle(box, outline=color, width=line_width)

            label_parts = []
            if show_class_names:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{score:.2f}")
            draw_detection_label(draw, box, " ".join(label_parts), color)

    return result_image, dict(counts), inference_ms, detected_objects


def get_available_model_names() -> list[str]:
    """Возвращает модели, для которых указан существующий файл весов."""
    return [
        model_name
        for model_name, model_path in MODEL_PATHS.items()
        if Path(model_path).is_file()
    ]


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Конвертирует PIL-изображение в bytes для st.download_button."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


# =============================================================================
# SIDEBAR — НАСТРОЙКИ
# =============================================================================

with st.sidebar:
    st.header("Настройки")

    available_models = get_available_model_names()
    if not available_models:
        st.error("Не найдено ни одного файла модели из MODEL_PATHS.")
        st.stop()

    selected_model = st.selectbox(
        "Модель",
        options=available_models,
        index=available_models.index("YOLOv12s") if "YOLOv12s" in available_models else 0,
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    show_confidence = st.checkbox("Показывать confidence", value=True)
    show_class_names = st.checkbox("Показывать названия классов", value=True)

    st.markdown(
        f"""
        <div class="sidebar-info">
            <strong>{selected_model}</strong><br><br>
            {MODEL_DESCRIPTIONS[selected_model]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ОСНОВНАЯ ОБЛАСТЬ — ЗАГОЛОВОК
# =============================================================================

st.title("🔍 Детекция изображения")
st.markdown(
    "Загрузите снимок ЛЭП, настройте параметры в боковой панели и запустите "
    "детекцию. Результат отобразится с разметкой найденных объектов и сводной "
    "таблицей по классам."
)

st.divider()


# =============================================================================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# =============================================================================

st.subheader("Загрузка изображения")

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"],
    help="Поддерживаемые форматы: JPG, JPEG, PNG",
)

run_detection_clicked = st.button(
    "Запустить детекцию",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)

if uploaded_file is None:
    st.info("Загрузите изображение, чтобы активировать кнопку детекции.")

st.divider()


# =============================================================================
# ДЕТЕКЦИЯ И ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

if uploaded_file is not None and run_detection_clicked:
    source_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner(f"Выполняется детекция ({selected_model})..."):
        result_image, detection_counts, inference_ms, detected_objects = run_detection(
            image=source_image,
            model_name=selected_model,
            conf=confidence_threshold,
            iou=iou_threshold,
            show_confidence=show_confidence,
            show_class_names=show_class_names,
        )

    history_id = save_detection(
        original_image=source_image,
        predicted_image=result_image,
        model_name=selected_model,
        detected_objects=detected_objects,
        image_name=uploaded_file.name,
    )

    st.subheader("Результаты")

    col_source, col_result = st.columns([1, 2])

    with col_source:
        st.image(source_image, use_container_width=True)
        st.caption("Исходное изображение")

    with col_result:
        st.image(result_image, use_container_width=True)
        st.caption("Результат детекции")

    st.divider()

    # --- Метрики ---
    total_detected = sum(detection_counts.values())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Используемая модель", selected_model)
    with metric_col2:
        st.metric("Время инференса", f"{inference_ms:.1f} мс")
    with metric_col3:
        st.metric("Количество найденных объектов", total_detected)

    st.caption(f"Запись сохранена в историю: {history_id}")

    st.divider()

    # --- Таблица обнаруженных объектов ---
    st.subheader("Обнаруженные объекты")

    detection_df = build_detection_table(detection_counts)
    styled_df = style_detection_table(detection_df)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Итог анализа ---
    st.subheader("Итог анализа")

    with st.container(border=True):
        st.markdown(build_summary_text(detection_counts))

    st.divider()

    # --- Скачивание результата ---
    # TODO: при необходимости добавить метаданные (JSON) рядом с изображением
    st.download_button(
        label="Скачать результат",
        data=image_to_bytes(result_image),
        file_name="detection_result.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )

elif uploaded_file is not None and not run_detection_clicked:
    st.warning("Изображение загружено. Нажмите «Запустить детекцию» для анализа.")
