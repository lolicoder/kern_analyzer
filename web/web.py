import reflex as rx  
from .pages import (view, segment_view, index, stretched_view, totall_view)


                        
app = rx.App()
app.add_page(index.index)
app.add_page(stretched_view.stretched_view, route='/stretched_view')
app.add_page(segment_view.segment_view, route='/segment')
app.add_page(view.view, route='/view')
app.add_page(totall_view.stretched_view, route='/totall_view')
