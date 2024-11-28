import reflex as rx
from ..state import State


color = "rgb(107,99,246)"

def segment_view():
    primary_color = "#6200ea"
    secondary_color = "#03dac5"
    background_color = "#f5f5f5"

    
    return rx.center(
        rx.box(
            rx.vstack(
                rx.text(
                    "Результат обработки",
                    font_size="2em",
                    font_weight="bold",
                    color=primary_color,
                    margin_bottom="1em",
                ),
                rx.box(
                    rx.image(
                        src=rx.get_upload_url(f"{State.selected_image['img_w_mask']}"),
                        width='90%',
                        border_radius="10px",
                        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                    ),
                    bg="white",
                    padding="1em",
                    border_radius="10px",
                    box_shadow="0 8px 16px rgba(0, 0, 0, 0.1)",
                    margin_bottom="2em",
                ),
                rx.cond(State.is_stretching,
                    rx.box(
                    rx.spinner(size='2',color='black'),
                    rx.text("Классификация и вытягивание...",
                            color='black'),
                    ),
                    ),

                rx.hstack(
                    rx.button(
                        "Вытянуть",
                        color="white",
                        bg=primary_color,
                        border_radius="10px",
                        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                        hover_bg="#3700b3",
                        padding="0.5em 1em",
                        on_click= State.handle_stretch(State.selected_image['path'], State.selected_image['msk_path']),
                    ),
                    rx.button(
                        "Продолжить",
                        color="white",
                        bg=secondary_color,
                        border_radius="10px",
                        box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                        hover_bg="#018786",
                        padding="0.5em 1em",
                        on_click=rx.redirect('/stretched_view'),
                    ),
                    spacing="1em",
                ),
            ),
            width=["95%", "80%", "70%"],
            padding="2em",
            bg="white",
            border_radius="10px",
            box_shadow="0 8px 16px rgba(0, 0, 0, 0.1)",
        ),
        bg=background_color,
        height="100vh",
    )