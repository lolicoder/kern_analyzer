import reflex as rx
from ..state import State

color = "rgb(107,99,246)"

def view():
    primary_color = "#6200ea"
    secondary_color = "#03dac5"
    background_color = "#f5f5f5"
    sidebar_bg = "#f9f9f9"

    return rx.center(
        rx.box(
            rx.hstack(
                rx.box(
                    rx.vstack(
                        rx.text(
                            "Список изображений",
                            font_size="1.5em",
                            font_weight="bold",
                            color=primary_color,
                            margin_bottom="1em",
                        ),
                        rx.foreach(
                            State.img_dict,
                            lambda image: rx.hstack(
                                rx.button(
                                    image['name'],
                                    width="100%",
                                    margin_bottom="0.5em",
                                    bg="white",
                                    color="black",
                                    on_click=lambda image=image: State.select_image(image),
                                    style={
                                        "_hover": {
                                            "bg": "#f0f0f0",
                                            "color": primary_color,
                                        },
                                        "border-radius": "0.5em",
                                        "padding": "0.5em 1em",
                                        "text-align": "left",
                                    },
                                ),
                                align_items="center",
                            )
                        ),
                        align_items="stretch",
                        spacing="1em",
                    ),
                    box_shadow="2px 0 8px rgba(0, 0, 0, 0.1)",
                    max_height="80vh",
                    overflow_y="auto",
                    padding="1em",
                    border="1px solid #ccc",
                    border_radius="10px",
                    bg=sidebar_bg,
                    width="25%",
                ),
                
                rx.box(
                    rx.vstack(
                        rx.text(
                            "Предпросмотр изображения",
                            font_size="2em",
                            font_weight="bold",
                            color=primary_color,
                            margin_bottom="1em",
                        ),
                        rx.box(
                            rx.image(
                                src=rx.get_upload_url(f'{State.selected_image["image"]}'),
                                width="100%",
                                border_radius="10px",
                                box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                                object_fit="contain",
                            ),
                            bg="white",
                            padding="1em",
                            border_radius="10px",
                            box_shadow="0 8px 16px rgba(0, 0, 0, 0.1)",
                            margin_bottom="2em",
                        ),
                        rx.cond(
                            State.is_segmenting,
                            rx.box(
                                rx.spinner(size="2", color=primary_color),
                                rx.text("Сегментация...", color=primary_color),
                            ),
                        ),
                        rx.hstack(
                            rx.button(
                                "Сегментировать",
                                color="white",
                                bg=primary_color,
                                border_radius="10px",
                                box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                                _hover={"bg": "#3700b3"},
                                padding="0.5em 1em",
                                on_click=lambda: State.handle_segment(State.selected_image['path']),
                            ),
                            rx.button(
                                "Продолжить",
                                color="white",
                                bg=secondary_color,
                                border_radius="10px",
                                box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                                _hover={"bg": "#018786"},
                                padding="0.5em 1em",
                                on_click=rx.redirect('/segment'),
                            ),
                            rx.button(
                                "Обработать все",
                                color="white",
                                bg="#03dac5",
                                border_radius="10px",
                                box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                                _hover={"bg": "#02c4a5"},
                                padding="0.5em 1em",
                                on_click=State.stretch_all(),
                            ),
                            rx.button(
                                "Перейти на страницу totall",
                                color="white",
                                bg=secondary_color,
                                border_radius="10px",
                                box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                                _hover={"bg": "#018786"},
                                padding="0.5em 1em",
                                on_click=rx.redirect('/totall_view'),
                            ),
                            spacing="1em",
                        ),
                    ),
                    width="70%",
                    padding="2em",
                    bg="white",
                    border_radius="10px",
                    box_shadow="0 8px 16px rgba(0, 0, 0, 0.1)",
                    margin_left="2em",
                ),
            ),
            bg=background_color,
            height="100vh",
            padding="2em",
        ),
    )
