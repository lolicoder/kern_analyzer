import reflex as rx
import numpy as np
from ..state import State

color = "rgb(107,99,246)"




def stretched_view():
    primary_color = "#6200ea"
    secondary_color = "#03dac5"
    background_color = "#f5f5f5"
    start_depth = State.depth_from
    step = 0.10 
    return rx.center(
        rx.box(
            rx.vstack(
                rx.text(
                    "Вытянутое изображение",
                    font_size="2em",
                    font_weight="bold",
                    color=primary_color,
                    margin_bottom="1em",
                ),
                rx.box(
                    rx.center(
                        rx.vstack(
                            rx.foreach(
                                State.img_parts,
                                lambda item, idx: rx.box(
                                    rx.hstack(
                                        rx.box(
                                            rx.text(
                                                rx.cond(
                                                    idx >= 0,
                                                    f"{start_depth + step * idx} м",
                                                    ""
                                                ),
                                                font_size="0.8em",
                                                text_align="right",
                                                color="gray",
                                                width="50px",
                                                padding="0",
                                            ),
                                            height="150px",
                                            border_right="1px solid #ccc",
                                            padding_right="0.5em",
                                            margin="0",
                                        ), 
                                        rx.tooltip(
                                            rx.box(
                                                rx.image(
                                                    src=rx.get_upload_url(f'{item["img"]}'),
                                                    width="150px",
                                                    height="150px",
                                                    hover_opacity="0.5",
                                                    transform="rotate(90deg)",
                                                ),
                                            ),
                                            content=f'{item['type'].to_string()}',
                                        ),
                                        rx.dialog.root(
                                                rx.dialog.trigger(rx.box(
                                                    bg=item['color'],
                                                    width="10px",
                                                    height="150px",
                                                    margin_left="5px",
                                                ),
                                            ),
                                            rx.dialog.content(
                                                rx.dialog.title("Изменить"),
                                                rx.dialog.description(
                                                    "This is a dialog component. You can render anything you want in here.",
                                                ),
                                                 rx.input(
                                                    value=f'{item['type'].to_string()}',
                                                    placeholder="Введите тип горной породы",
                                                ),
                                                rx.dialog.close(
                                                    rx.button("Сохранить изменения", size="1"),
                                                ),
                                            ),
                                            ),
                                        # rx.tooltip(
                                        #     rx.box(
                                        #         bg=item['color'],
                                        #         width="10px",
                                        #         height="150px",
                                        #         margin_left="5px",
                                        #     ),
                                        #     content=f'{item['type'].to_string()}',
                                        # ),
                                    ),
                                    margin_bottom="0em",
                                    border_bottom="1px solid #ccc"
                                ),
                            ),
                            spacing="0em",
                        ),
                    ),
                    max_height="600px",
                    overflow_y="auto",
                    padding="1em",
                    border="1px solid #ccc",
                    border_radius="10px",
                    style={
                        "scrollbar-width": "thin",
                        "scrollbar-color": "#ccc transparent",
                    },
                ),
                rx.button(
                    "Вернуться",
                    color="white",
                    bg=secondary_color,
                    border_radius="10px",
                    box_shadow="0 4px 6px rgba(0, 0, 0, 0.1)",
                    hover_bg="#018786",
                    padding="0.5em 1em",
                    on_click=rx.redirect('/'),
                    margin_top="2em",
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