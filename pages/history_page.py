from __future__ import annotations

import pandas as pd
import streamlit as st

from history_manager import (
    clear_history,
    export_history,
    get_detection,
    load_history,
    resolve_history_path,
)


# СТИЛИ
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .history-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: inherit;
        margin-bottom: 0.35rem;
        word-break: break-word;
    }
    .history-meta {
        color: inherit;
        opacity: 0.78;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .empty-history {
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ДАННЫЕ И ФИЛЬТРЫ
def format_history_date(value: pd.Timestamp) -> str:

    if pd.isna(value):
        return "-"
    return value.strftime("%d.%m.%Y %H:%M")


def get_history_size_label(history: pd.DataFrame) -> str:
    
    total_bytes = 0
    path_columns = ["original_path", "predicted_path", "result_path"]

    for _, row in history.iterrows():
        for column in path_columns:
            path_value = row.get(column)
            if not isinstance(path_value, str) or not path_value:
                continue
            path = resolve_history_path(path_value)
            if path.is_file():
                total_bytes += path.stat().st_size

    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} КБ"
    return f"{total_bytes / (1024 * 1024):.1f} МБ"


def show_filters(history: pd.DataFrame) -> tuple[str, str, str]:

    model_options = ["Все"] + sorted(
        model for model in history["model"].dropna().unique().tolist() if model
    )

    with st.container(border=True):
        filter_col1, filter_col2, filter_col3 = st.columns([1.1, 2.2, 1.4])

        with filter_col1:
            selected_model = st.selectbox(
                "Модель",
                options=model_options,
                index=0,
                key="history_filter_model",
            )

        with filter_col2:
            search_query = st.text_input(
                "Поиск по имени файла",
                placeholder="Например: line_42_segment_08.jpg",
                key="history_search_filename",
            )

        with filter_col3:
            sort_order = st.selectbox(
                "Сортировка",
                options=["Сначала новые", "Сначала старые", "Больше объектов", "Меньше объектов"],
                index=0,
                key="history_sort_order",
            )

    return selected_model, search_query, sort_order


def apply_filters(
    history: pd.DataFrame,
    selected_model: str,
    search_query: str,
    sort_order: str,
) -> pd.DataFrame:

    filtered = history.copy()

    if selected_model != "Все":
        filtered = filtered[filtered["model"] == selected_model]

    search_query = search_query.strip().lower()
    if search_query:
        filtered = filtered[
            filtered["image_name"].fillna("").str.lower().str.contains(search_query, regex=False)
        ]

    if sort_order == "Сначала новые":
        filtered = filtered.sort_values("date", ascending=False, na_position="last")
    elif sort_order == "Сначала старые":
        filtered = filtered.sort_values("date", ascending=True, na_position="last")
    elif sort_order == "Больше объектов":
        filtered = filtered.sort_values("objects_count", ascending=False)
    elif sort_order == "Меньше объектов":
        filtered = filtered.sort_values("objects_count", ascending=True)

    return filtered



# UI-БЛОКИ
def build_detection_table(detection: dict) -> pd.DataFrame:
    
    rows = []
    for obj in detection.get("objects", []):
        rows.append(
            {
                "Класс": obj.get("class", ""),
                "confidence": round(float(obj.get("confidence", 0.0)), 3),
                "bbox": obj.get("bbox", []),
            }
        )
    return pd.DataFrame(rows, columns=["Класс", "confidence", "bbox"])


def show_detection_details(record: pd.Series) -> None:

    detection = get_detection(str(record["id"]))
    predicted_path = resolve_history_path(str(record["predicted_path"]))

    with st.expander("Подробности детекции", expanded=True):
        image_col, table_col = st.columns([1.2, 1])

        with image_col:
            if predicted_path.is_file():
                st.image(str(predicted_path), use_container_width=True)
            else:
                st.warning("Изображение после детекции не найдено.")

        with table_col:
            st.markdown("##### Найденные объекты")
            st.dataframe(
                build_detection_table(detection),
                use_container_width=True,
                hide_index=True,
            )


def show_history_cards(history: pd.DataFrame) -> None:

    if "selected_detection_id" not in st.session_state:
        st.session_state.selected_detection_id = None

    for _, record in history.iterrows():
        original_path = resolve_history_path(str(record["original_path"]))

        with st.container(border=True):
            image_col, info_col, action_col = st.columns([1.1, 4, 1.1])

            with image_col:
                if original_path.is_file():
                    st.image(str(original_path), use_container_width=True)
                else:
                    st.info("Нет изображения")

            with info_col:
                st.markdown(
                    f"""
                    <div class="history-title">{record["image_name"]}</div>
                    <div class="history-meta">
                        Дата: <strong>{format_history_date(record["date"])}</strong><br>
                        Модель: <strong>{record["model"]}</strong><br>
                        Найдено объектов: <strong>{int(record["objects_count"])}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with action_col:
                st.write("")
                st.write("")
                if st.button("Подробнее", key=f"details_{record['id']}", use_container_width=True):
                    st.session_state.selected_detection_id = record["id"]

            if st.session_state.selected_detection_id == record["id"]:
                show_detection_details(record)


def show_sidebar(history: pd.DataFrame) -> None:

    with st.sidebar:
        st.header("Информация")

        st.metric("Всего сохранено изображений", len(history))
        st.metric("Всего моделей", history["model"].nunique() if not history.empty else 0)
        st.metric("Размер истории", get_history_size_label(history))

        st.divider()
        st.subheader("Действия")

        st.download_button(
            "Экспорт CSV",
            data=export_history(),
            file_name="history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        confirm_clear = st.checkbox("Подтвердить очистку истории")
        if st.button("Очистить историю", use_container_width=True, disabled=not confirm_clear):
            clear_history()
            st.session_state.selected_detection_id = None
            st.success("История очищена.")
            st.rerun()


def show_empty_history() -> None:

    st.markdown(
        """
        <div class="empty-history">
            <h3>История детекций пуста.</h3>
            <p>Загрузите изображение на странице "Детекция".</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:

    st.title("История детекций")
    st.markdown("На данной странице отображаются все выполненные детекции изображений.")


# =============================================================================
# ОСНОВНАЯ СТРАНИЦА
# =============================================================================

def main() -> None:

    history = load_history()
    show_sidebar(history)
    show_header()

    if history.empty:
        show_empty_history()
        return

    selected_model, search_query, sort_order = show_filters(history)
    filtered_history = apply_filters(history, selected_model, search_query, sort_order)

    st.subheader("История")
    if filtered_history.empty:
        st.info("По выбранным фильтрам записей не найдено.")
        return

    show_history_cards(filtered_history)


main()
