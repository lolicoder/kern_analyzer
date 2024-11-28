import reflex as rx
from pathlib import Path
import os
import re
import shutil
import numpy as np
from PIL import Image
import model.segmentation_model as SM

class State(rx.State):
    UPLOAD_DIR = Path('uploaded_files')
    MODEL_PATH = Path('../deeplabv3_rock_segmentation.pth')
    
    image = []
    uploaded_count = 0
    file_names = []
    totall_slices = []
    depth_from: int
    depth_to: int
    totall_depth_from = 0
    totall_depth_to = 0
    img_parts: list[dict] = [] 
    totall_parts : list[dict] = []
    is_loading: bool = False
    is_segmenting: bool = False  
    is_stretching: bool = False
    selected_image = {}
    img_dict : list[dict] = []
    colors = {
        '107': 'lightblue',     
        '113': 'lightgreen',    
        '122': 'lightcoral',    
        '127': 'khaki',         
        '128': 'plum',          
        '140': 'peachpuff',     
        '142': 'lavender',      
    }
    
    def natural_key(self, string):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]

    def classify(self, file_name):
        slices_dir = self.get_slices_dir(file_name)
        slices = sorted([p.name for p in slices_dir.iterdir() if p.is_file()], key=self.natural_key)
        self.img_parts = []
        for img in slices[:-2]:
            slice_path1 = slices_dir / img
            slice_path = Path(file_name) / 'slices' / img
            pred_class = SM.predict_and_visualize(slice_path1)
            self.selected_image['slices'].append({
                'img': str(slice_path), 
                'type': str(pred_class), 
                'color': self.colors.get(str(pred_class), 'white')
            })
            self.img_parts.append({
                'img': str(slice_path), 
                'type': str(pred_class), 
                'color': self.colors.get(str(pred_class), 'white')
            })
    def change_type(self, img, value):
        img['type'] = value


    def get_image_dir(self, file_name):
        return self.UPLOAD_DIR / file_name

    def get_slices_dir(self, file_name):
        return self.get_image_dir(file_name) / 'slices'

    def get_mask_path(self, file_name):
        return self.get_image_dir(file_name) / 'seg_mask.jpg'

    def get_image_with_mask_path(self, file_name):
        return self.get_image_dir(file_name) / 'image_w_mask.jpg'
    
    

    @rx.event
    def select_image(self, image: str):
        self.selected_image = image  

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        self.is_loading = True
        self.file_names = []
        self.img_dict = []
        self.uploaded_count = 0

        for file in files:
            upload_data = await file.read()
            file_name = file.filename[:-4]
            image_dir = self.get_image_dir(file_name)
            image_dir.mkdir(parents=True, exist_ok=True)
            outfile = image_dir / file.filename
            image_path = outfile

            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            self.uploaded_count = len(files)
            self.img_dict.append({
                'image': str(Path(file_name) / file.filename), 
                'path': str(image_path),
                'name': file.filename,
                'img_w_mask': '',
                'msk_path': '',
                'slices': [],
                'is_segmented': False,
                'is_stretched': False
            })
            self.selected_image = self.img_dict[0]
            self.file_names.append(file_name)
       

        self.is_loading = False

    @rx.event
    async def handle_segment(self, image_path):
        self.is_segmenting = True
        yield

        model = SM.load_segmentation_model(str(self.MODEL_PATH))
        msk_path = SM.segment(model, image_path)
        SM.show_masks_on_image_one(image_path, msk_path)

        file_name = self.selected_image['name'][:-4]
        self.selected_image['msk_path'] = str(self.get_mask_path(file_name))
        self.selected_image['img_w_mask'] = str(Path(file_name) / 'image_w_mask.jpg')
        self.selected_image['is_segmented'] = True

        yield
        self.is_segmenting = False

    @rx.event
    async def handle_stretch(self, image_path, mask_path):
        self.is_stretching = True
        yield

        strip = SM.proper_longer(mask_path, image_path)
        file_name = Path(image_path).stem
        self.depth_from, self.depth_to = SM.parse_depth_from_filename(file_name)
        slices_dir = self.get_slices_dir(file_name)
        SM.proper_slicer(strip, self.depth_from, self.depth_to, str(slices_dir))
        self.classify(file_name)
        self.selected_image['is_stretched'] = True

        yield
        self.is_stretching = False
        print('stretched')

    @rx.event
    async def stretch_all(self):
        totall_dir = self.UPLOAD_DIR / 'totall'
        if totall_dir.exists():
            shutil.rmtree(totall_dir)
        totall_dir.mkdir(parents=True, exist_ok=True)

        self.totall_slices = []
        images = sorted(self.file_names, key=self.natural_key)
        print(f'\n{images}\n')
        self.totall_depth_from, _ = SM.parse_depth_from_filename(self.img_dict[0]['name'])
        _, self.totall_depth_to = SM.parse_depth_from_filename(self.img_dict[-1]['name'])

        model = SM.load_segmentation_model(str(self.MODEL_PATH))

        for image in self.img_dict:
            self.selected_image = image
            print(self.selected_image)
            image_path = self.selected_image['path']
            print(image_path)
            print('\n')
            file_name = self.selected_image['name'][:-4]
            SM.segment(model, image_path)

            self.selected_image['is_segmented'] = True
            self.selected_image['msk_path'] = str(self.get_mask_path(file_name))
            msk_path = self.selected_image['msk_path']
            print(f'{msk_path}\n')

            strip = SM.proper_longer(msk_path, image_path)
            print(f'изображение {file_name} обработано\n')

        print('готово\n')  
        strips = []
        for i in images:
            strip_path = self.UPLOAD_DIR / i / 'strip.jpg'
            strip = Image.open(strip_path)
            strips.append(strip)
            print(strip_path)

        total_width = sum([strip.width for strip in strips])
        max_height = max([strip.height for strip in strips])

        totall_strip = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for strip in strips:
            y_offset = (max_height - strip.height) // 2
            totall_strip.paste(strip, (x_offset, y_offset))
            x_offset += strip.width

        totall_strip_path = totall_dir / 'totall_strip.png'
        totall_strip.save(totall_strip_path)
        print(f'Общий strip сохранен в {totall_strip_path}\n')

        slices_dir = totall_dir / 'slices'
        self.totall_slices = SM.proper_slicer(np.array(totall_strip), self.totall_depth_from, self.totall_depth_to, str(slices_dir))  
        print('слайсы готовы')

        slices = sorted([p.name for p in slices_dir.iterdir() if p.is_file()], key=self.natural_key)
        self.totall_parts = []
        for img in slices[:-2]:
            slice_path = slices_dir / img
            slice_path1 = Path('totall') / 'slices' / img
            pred_class = SM.predict_and_visualize(slice_path)

            self.totall_parts.append({
                'img': str(slice_path1), 
                'type': str(pred_class), 
                'color': self.colors.get(str(pred_class), 'white')
            })
        print('обработано')