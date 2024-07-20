from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import pytesseract
import json

class OCR_Check(BaseModel):
    image: UploadFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],  # This allows all HTTP methods
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

def scan_document(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Loop over contours to find the document
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            doc_contour = approx
            break
    
    # Apply perspective transform to obtain the top-down view of the document
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    scanned = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return scanned

def get_text(image):
    return pytesseract.image_to_string(image)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "ocr_endpoint": "http://127.0.0.1:8000/ocr"})

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    scanned_image = scan_document(image)
    is_success, buffer = cv2.imencode(".jpg", scanned_image)
    io_buf = io.BytesIO(buffer)
    return StreamingResponse(io_buf, media_type="image/jpeg")

@app.post("/ocr")
async def ocr(data: UploadFile = File(...)):
    contents = await data.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    scanned_image = scan_document(image)
    text = get_text(scanned_image)
    return Response(json.dumps({"data": text}), media_type="application/json")
