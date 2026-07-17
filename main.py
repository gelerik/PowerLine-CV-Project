import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="Power Line Defect Detection API")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


FAST_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo12n.pt")
MIDDLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo12s.pt")
ACCURATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo12L.pt")

models = {}

def get_model(model_type: str):
    if model_type not in models:
        print(f"⏳ Загрузка модели '{model_type}' в память... Подождите...")
        
        path_to_load = FAST_MODEL_PATH if model_type == "fast" else ACCURATE_MODEL_PATH
        if not os.path.exists(path_to_load):
            raise FileNotFoundError(f"Файл модели не найден по пути: {path_to_load}")

        if model_type == "fast":
            models["fast"] = YOLO(FAST_MODEL_PATH)
        elif model_type == "accurate":
            models["accurate"] = YOLO(ACCURATE_MODEL_PATH)
            
        print(f"✅ Модель '{model_type}' успешно загружена!")
        
    return models[model_type]

@app.get("/")
def read_root():
    return {"message": "API is running! Send POST request to /predict"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    model_type: str = Form("fast") 
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if model_type not in ["fast", "accurate"]:
            return JSONResponse(status_code=400, content={"error": "Используйте 'fast' или 'accurate'"})
        
        selected_model = get_model(model_type)
        
        results = selected_model.predict(image, conf=0.25)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": selected_model.names[int(box.cls)], 
                    "confidence": float(box.conf),                    
                    "box": [float(x) for x in box.xyxy[0]]            
                })
                
        return {"model_used": model_type, "detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
