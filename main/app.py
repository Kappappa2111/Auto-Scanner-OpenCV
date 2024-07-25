from fastapi import FastAPI, Request, File, UploadFile, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
from main import scan_and_generate_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_path = scan_and_generate_pdf(contents)

    if os.path.exists(pdf_path):
        return HTMLResponse(content=f"<p>File uploaded successfully! <a href='/download/{os.path.basename(pdf_path)}'>Download PDF</a></p>")
    else:
        return Response("Error processing image", status_code=500)

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join("Scanned_PDF", file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=file_name)
    else:
        return Response("File not found", status_code=404)
