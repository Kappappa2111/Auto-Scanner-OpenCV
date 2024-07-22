from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import numpy as np
from PIL import Image
from main import scan_document, get_text
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Điều chỉnh cái này theo nhu cầu của bạn
    allow_credentials=True,
    allow_methods=["*"],  # Điều này cho phép tất cả các phương thức HTTP
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    scanned_image = scan_document(image)
    is_success, buffer = cv2.imencode(".jpg", scanned_image)
    file_name = f"{uuid.uuid4()}.jpg"
    file_path = f"scanned_images/{file_name}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(buffer)

    return {"download_url": f"/download/{file_name}"}

@app.get("/download/{file_name}", response_class=HTMLResponse)
async def download_file(file_name: str):
    file_path = f"scanned_images/{file_name}"
    return StreamingResponse(io.BytesIO(open(file_path, "rb").read()), media_type="image/jpeg")

@app.post("/ocr")
async def ocr(data: UploadFile = File(...)):
    contents = await data.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    scanned_image = scan_document(image)
    text = get_text(scanned_image)
    return Response(json.dumps({"data": text}), media_type="application/json")
