import utils
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Request
import os
from os import getcwd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTasks
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory = "static"), name = "static")
templates = Jinja2Templates(directory = "templates")

PATH_FILES = getcwd() + "/"

def cleanup(temppath, f):
    if True:
        time.sleep(2)
        os.remove(os.path.join(temppath, f))

@app.get("/")
# def read_root():
#     return {"msg": "Hello World"}
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def predict(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # with open(f'{PATH_FILES}{file.filename}', "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    with open(PATH_FILES + file.filename, "wb") as myfile:
        content = await file.read()
        myfile.write(content)
        myfile.close()
    print(type(file.filename))
    results = utils.colorize(imagename = file.filename)
    background_tasks.add_task(cleanup, PATH_FILES, file.filename)
    return templates.TemplateResponse("index.html", {"request": request, "results": results})

