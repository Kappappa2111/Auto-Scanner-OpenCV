from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from main import *
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],  # This allows all HTTP methods
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,"ocr_endpoint": " http://127.0.0.1:8000/ocr"})
@app.post("/ocr")
async def ocr(data:ORC_Check):
    #rgb image pil image
    image = scan_document(data.image)
    text  = get_text(image)
    return Response(json.dumps({"data":text}))