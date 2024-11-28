import reflex as rx
from ..state import State

color = "rgb(107,99,246)"

def index():
    primary_color = "#6200ea"
    secondary_color = "#03dac5"
    background_color = "#f5f5f5"
    
    return rx.center( 
        rx.box(
            rx.vstack(
                rx.text(
                    "Загрузите файл для анализа",
                    font_size="2em",
                    font_weight="bold",
                    color=primary_color,
                    margin_bottom="1em",
                ),
                rx.upload(
                    rx.vstack(
                        rx.button(
                            "Выберите файл",
                            color=primary_color,
                            bg="white",
                            border=f"1px solid {primary_color}",
                            border_radius="10px",
                            padding="0.5em 1em",
                            box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                            hover_bg="#3700b3",
                            hover_color="white",
                        ),
                        rx.text(
                            "Или перетащите файл сюда",
                            color="#777",
                            font_size="0.9em",
                            margin_top="0.5em",
                        ),
                    ),
                    id="upload1",
                    border=f"1px dashed {primary_color}",
                    padding="2em",
                    border_radius="10px",
                    width=["90%", "60%", "40%"],
                    margin_bottom="2em",
                ),
                rx.cond(
                    State.is_loading,
                    rx.box(
                    rx.spinner(size='2',color='black'),
                    rx.text("загрузка...",
                            color='black'),
                    ),
                ),
                rx.cond(
                    State.uploaded_count>0,
                    rx.text(
                    f"Загружено файлов: {State.uploaded_count}",
                    font_size="1em",
                    color='black',
                    margin_top="1em",
                )
                ),
                rx.hstack(
                    rx.button(
                        "Загрузить",
                        color="white",
                        bg=primary_color,
                        border_radius="10px",
                        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                        hover_bg="#3700b3",
                        padding="0.5em 1em",
                        on_click=State.handle_upload(
                            rx.upload_files(upload_id="upload1")
                        ),
                    ),
                    rx.button(
                        "Продолжить",
                        color="white",
                        bg=secondary_color,
                        border_radius="10px",
                        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                        hover_bg="#018786",
                        padding="0.5em 1em",
                        on_click=rx.redirect('/view'),
                    ),
                    spacing="1em",
                ),
            ),
            width=["90%", "70%", "50%"],
            padding="2em",
            bg="white",
            border_radius="10px",
            box_shadow="0 8px 16px rgba(0, 0, 0, 0.1)",
        ),
        bg=background_color,
        height="100vh",
    )