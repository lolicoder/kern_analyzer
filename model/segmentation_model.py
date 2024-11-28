import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
import torch
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights   
from torchvision.models import ResNet50_Weights

def parse_depth_from_filename(filename):
    
    match = re.search(r'_(\d+,\d+)-(\d+,\d+)', filename)
    if match:
        depth_from = float(match.group(1).replace(',', '.'))
        depth_to = float(match.group(2).replace(',', '.'))
        return depth_from, depth_to
    else:
        return 0, 100

def proper_slicer(strip, depth_from, depth_to, output_folder='output_slices', step=100):
   
    

    slices=[]
    horizontal_image = strip
    depth_start_mm = depth_from * 1000
    depth_end_mm = depth_to * 1000
    slice_interval_mm = step
    image_width = horizontal_image.shape[1]
    mm_per_pixel = (depth_end_mm - depth_start_mm) / image_width

    def calculate_depth_range(slice_index, interval_mm, mm_per_pixel):
        depth_start = depth_from + (slice_index * interval_mm / 1000)
        depth_end = depth_start + (interval_mm / 1000)
        return depth_start, depth_end

    
    slice_left = 0
    slice_right = int(slice_interval_mm / mm_per_pixel)
    n_slice = 0

    
    os.makedirs(output_folder, exist_ok=True)

    while slice_left < image_width:
        depth_range = calculate_depth_range(n_slice, slice_interval_mm, mm_per_pixel)
        

        
        slice_img = horizontal_image[:, slice_left:slice_right]
        

        
        slice_filename = f'slice_{n_slice}_{depth_range[0]:.2f}-{depth_range[1]:.2f}.png'
        slice_path = os.path.join(output_folder, slice_filename)
        plt.imsave(slice_path, slice_img)
        slices.append(slice_img)
       
        n_slice += 1
        slice_left = slice_right
        slice_right += int(slice_interval_mm / mm_per_pixel)
    return slices    

def load_classification_model():
    
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  
    num_classes = 7
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(torch.load('../model_resnet_100_epoch_weights.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()  
    return model

def predict_and_visualize(image_path):

    model = load_classification_model()

    transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])

    
    class_names = ['107', '113', '122', '127', '128', '140', '142']
    
    
    image_rgb = Image.open(image_path).convert("RGB")
    

    
    input_image = transform(image_rgb)
    input_image = input_image.unsqueeze(0)  
    input_image = input_image

    
    model.eval()

   
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    
    return predicted_class

def proper_longer(mask_path, raw_image_path):
    raw_image = np.array(Image.open(raw_image_path))
    mask_np = np.array(Image.open(mask_path).convert("L")) 
    threshold = 127  
    mask_np = (np.array(mask_np) > threshold).astype(np.uint8) * 255
      

    larger_kernel = np.ones((10, 10), np.uint8)
    eroded_image_strong = cv2.erode(mask_np, larger_kernel, iterations=3)

    
    if eroded_image_strong.dtype != np.uint8:
        eroded_image_strong = eroded_image_strong.astype(np.uint8)

    
    dist_transform_strong = cv2.distanceTransform(eroded_image_strong, cv2.DIST_L2, 5)
    _, sure_fg_strong = cv2.threshold(dist_transform_strong, 0.4 * dist_transform_strong.max(), 255, 0)

    sure_fg_strong = np.uint8(sure_fg_strong)
    sure_bg_strong = cv2.dilate(eroded_image_strong, larger_kernel, iterations=3)
    unknown_strong = cv2.subtract(sure_bg_strong, sure_fg_strong)

    _, markers_strong = cv2.connectedComponents(sure_fg_strong)
    markers_strong = markers_strong + 1
    markers_strong[unknown_strong == 255] = 0

    image_color_strong = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    markers_watershed_strong = cv2.watershed(image_color_strong, markers_strong)
    image_color_strong[markers_watershed_strong == -1] = [255, 0, 0]

    contours_watershed_strong, _ = cv2.findContours((markers_watershed_strong > 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours = sorted(contours_watershed_strong, key=lambda c: cv2.boundingRect(c)[1])

    rows = []
    current_row = []
    last_y = None
    tolerance = 30

    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if last_y is None or abs(y - last_y) < tolerance:
            current_row.append((x, y, w, h, contour))
        else:
            rows.append(current_row)
            current_row = [(x, y, w, h, contour)]
        last_y = y

    if current_row:
        rows.append(current_row)

    bounding_boxes = []
    for row in rows:
        min_x = min([x for x, y, w, h, contour in row])
        max_x = max([x + w for x, y, w, h, contour in row])
        min_y = min([y for x, y, w, h, contour in row])
        max_y = max([y + h for x, y, w, h, contour in row])

        bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

    width_sum = sum([box[2] for box in bounding_boxes])
    height_max = max([box[3] for box in bounding_boxes])
    combined_strip_horizontal = np.zeros((height_max, width_sum, 3), dtype=np.uint8)

    current_x = 0
    for x, y, w, h in bounding_boxes:
        strip = raw_image[y:y+h, x:x+w]
        combined_strip_horizontal[0:h, current_x:current_x+w] = strip
        current_x += w

    image_dir = os.path.dirname(raw_image_path)    
    plt.imsave(f'{image_dir}/strip.jpg', combined_strip_horizontal)

    return combined_strip_horizontal
    
 
def load_segmentation_model(model_path, device='cpu'):
    
    model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT )
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
   
    model.eval()  
    return model

def show_masks_on_image_one(raw_image_path, mask_path):
    image = np.array(Image.open(raw_image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))
    mask_np = (mask > 127).astype(np.uint8) * 255
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[:, :, 2] = mask_np  

    alpha = 0.4  
   
    image = image.astype(np.float32)
    colored_mask = colored_mask.astype(np.float32)

    alpha_mask = (mask_np > 0).astype(np.float32) * alpha

    overlay_image = image.copy()

    for c in range(3):
        overlay_image[:, :, c] = (1 - alpha_mask) * overlay_image[:, :, c] + alpha_mask * colored_mask[:, :, c]

    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

    image_dir = os.path.dirname(raw_image_path)
    plt.imsave(f'{image_dir}/image_w_mask.jpg', overlay_image.astype(np.uint8))

   



def segment(model, image_path, device='cpu'):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size 
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),         
    ])

    image_dir = os.path.dirname(image_path)
   
    input_image = transform(image).unsqueeze(0) 

    
    input_image = input_image.to(device)

    
    model.eval()
    with torch.no_grad():
        output = model(input_image)['out'] 
        pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  

    
    pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(original_size, resample=Image.NEAREST)

    plt.imsave(f'{image_dir}/seg_mask.jpg', pred_mask_resized)

    return f'{image_dir}/seg_mask.jpg'