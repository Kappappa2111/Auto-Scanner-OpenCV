from PIL import Image
import numpy as np
import cv2
from skimage.filters import threshold_local
from io import BytesIO
import os
import datetime
import time
from pathlib import Path

def scan_document(image):
    img = Image.fromarray(image)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    # Process image
    imgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = threshold_local(imgr, 17, offset=15, method='gaussian')
    imgr = (imgr > t).astype('uint8') * 255
    
    return imgr

def create_pdf(image_list):
    pdf_c = 0
    repn = Path('Scanned_PDF')
    if not repn.is_dir():
        os.mkdir('Scanned_PDF')

    pdf_images = [Image.open(img).convert('RGB') for img in image_list]
    ts = time.time()
    timeStam = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStam.split(":")
    pdf_path = f'./Scanned_PDF/Scanned_{pdf_c}_{Hour}_{Minute}_{Second}.pdf'
    pdf_images[0].save(pdf_path, save_all=True, append_images=pdf_images[1:])

    return pdf_path

def scan_and_generate_pdf(image_bytes):
    # Convert bytes to image
    image = Image.open(BytesIO(image_bytes))  # Ensure PIL is imported correctly
    scanned_image = scan_document(np.array(image))

    # Save scanned image temporarily
    temp_img_path = 'temp_scanned_image.jpg'
    cv2.imwrite(temp_img_path, scanned_image)
    
    # Generate PDF
    pdf_path = create_pdf([temp_img_path])
    
    # Clean up
    os.remove(temp_img_path)
    
    return pdf_path