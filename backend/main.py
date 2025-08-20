from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import io
from PIL import Image
import base64
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import asyncio
import traceback
import uvicorn
import os
from databases import Database
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import pathlib

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DATABASE SETUP ----------
DATABASE_URL = os.environ.get('DATABASE_URL')

# Fix old-style URLs from Render
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Ensure SSL is required
if DATABASE_URL and 'sslmode' not in DATABASE_URL:
    if '?' in DATABASE_URL:
        DATABASE_URL += '&sslmode=require'
    else:
        DATABASE_URL += '?sslmode=require'

# Initialize database
database = Database(DATABASE_URL if DATABASE_URL else 'postgresql://postgres:mamerto@localhost:5432/rice_yield')

# ---------- CONSTANTS ----------
RICE_VARIETY_FACTORS = {
    "Jasmine": 1.0,
    "Basmati": 0.95,
    "Arborio": 1.1,
    "Calrose": 1.05,
    "Japonica": 0.98,
    "Indica": 1.02
}

FERTILIZER_EFFECTIVENESS = {
    "Urea": 0.9,
    "DAP": 1.0,
    "NPK": 1.15,
    "Organic": 0.85
}

# ---------- MODEL ----------
model = None

# ---------- DATABASE FUNCTIONS ----------
async def init_db():
    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            await database.connect()
            await database.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    width REAL,
                    height REAL,
                    healthy_area REAL,
                    medium_area REAL,
                    unhealthy_area REAL,
                    predicted_yield REAL,
                    actual_yield REAL,
                    location TEXT,
                    model_type TEXT,
                    rice_variety TEXT,
                    fertilizer_type TEXT
                )
                """
            )
            print("Database connected successfully")
            return True
        except Exception as e:
            print(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Max retries exceeded. Database connection failed.")
                return False

async def train_model():
    try:
        print("Training yield prediction model...")
        rows = await database.fetch_all("SELECT * FROM historical_data WHERE actual_yield IS NOT NULL")
        df = pd.DataFrame([dict(r) for r in rows])
        if df.empty or len(df) < 10:
            print("Insufficient data for training (min 10 records)")
            return False

        df['rice_variety'] = df.get('rice_variety', 'Jasmine')
        df['fertilizer_type'] = df.get('fertilizer_type', 'Urea')
        df['variety_factor'] = df['rice_variety'].map(RICE_VARIETY_FACTORS)
        df['fertilizer_factor'] = df['fertilizer_type'].map(FERTILIZER_EFFECTIVENESS)
        df.fillna({'variety_factor': 1.0, 'fertilizer_factor': 1.0}, inplace=True)

        X = df[['healthy_area', 'medium_area', 'unhealthy_area', 'width', 'height', 'variety_factor', 'fertilizer_factor']]
        y = df['actual_yield']

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        joblib.dump(rf_model, 'yield_model.pkl')
        print(f"Model trained with {len(df)} records")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return False

# ---------- IMAGE FUNCTIONS ----------
def is_valid_image_type(content_type: str, filename: str) -> bool:
    valid_types = ["image/jpeg", "image/jpg", "image/png"]
    valid_extensions = [".jpg", ".jpeg", ".png"]
    return content_type.lower() in valid_types or any(filename.lower().endswith(ext) for ext in valid_extensions)

def is_rice_field(image_np: np.ndarray) -> bool:
    try:
        hsv = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.count_nonzero(green_mask) / (image_np.shape[0] * image_np.shape[1]) * 100
        return green_percentage > 30
    except Exception as e:
        print(f"Rice field detection error: {e}")
        traceback.print_exc()
        return False

def process_image(image_data: bytes, width_m: float, height_m: float, rice_variety: str, fertilizer_type: str):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(image).astype(np.float32)
        if not is_rice_field(img_np):
            return {"error": "Uploaded image does not appear to be a rice field."}

        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        pseudo_ndvi = (G - R) / (G + R + 1e-6)
        mask_healthy = pseudo_ndvi > 0.2
        mask_medium = (pseudo_ndvi <= 0.2) & (pseudo_ndvi > 0)
        mask_unhealthy = pseudo_ndvi <= 0

        out_img = np.zeros_like(img_np)
        out_img[mask_healthy] = [0, 255, 0]
        out_img[mask_medium] = [255, 255, 0]
        out_img[mask_unhealthy] = [255, 0, 0]

        total_pixels = img_np.shape[0] * img_np.shape[1]
        m2_per_pixel = (width_m * height_m) / total_pixels
        healthy_area = np.sum(mask_healthy) * m2_per_pixel
        medium_area = np.sum(mask_medium) * m2_per_pixel
        unhealthy_area = np.sum(mask_unhealthy) * m2_per_pixel

        variety_factor = RICE_VARIETY_FACTORS.get(rice_variety, 1.0)
        fertilizer_factor = FERTILIZER_EFFECTIVENESS.get(fertilizer_type, 1.0)

        global model
        model_type = "linear"
        if model:
            input_data = [[healthy_area, medium_area, unhealthy_area, width_m, height_m, variety_factor, fertilizer_factor]]
            yield_kg = model.predict(input_data)[0]
            model_type = "ML"
        else:
            base_yield = (healthy_area * 0.8) + (medium_area * 0.4) + (unhealthy_area * 0.1)
            yield_kg = base_yield * variety_factor * fertilizer_factor

        out_pil = Image.fromarray(out_img.astype(np.uint8))
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        processed_image_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "processed_image": processed_image_b64,
            "estimated_yield": yield_kg,
            "model_type": model_type,
            "stats": {
                "healthy": round(healthy_area, 2),
                "medium": round(medium_area, 2),
                "unhealthy": round(unhealthy_area, 2)
            },
            "fertilizer_type": fertilizer_type
        }
    except Exception as e:
        print(f"Image processing error: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# ---------- Pydantic Models ----------
class UpdateFertilizerRequest(BaseModel):
    record_id: int
    fertilizer_type: str
    rice_variety: str
    healthy_area: float
    medium_area: float
    unhealthy_area: float
    width: float
    height: float

class SaveActualYieldRequest(BaseModel):
    actualYield: float
    record_id: int
    rice_variety: str
    fertilizer_type: str

# ---------- API ENDPOINTS ----------
@app.get("/health")
async def health_check():
    try:
        await database.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {e}"
    return {"status": "ok", "database": db_status, "timestamp": datetime.now().isoformat()}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    width: float = Form(...),
    height: float = Form(...),
    rice_variety: str = Form(...),
    fertilizer_type: str = Form(...),
    location: Optional[str] = Form(None)
):
    try:
        if not is_valid_image_type(image.content_type, image.filename):
            return JSONResponse(status_code=400, content={"error": "Only JPG, JPEG, PNG allowed"})
        contents = await image.read()
        result = process_image(contents, width, height, rice_variety, fertilizer_type)
        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})

        if database.is_connected:
            query = """
            INSERT INTO historical_data (width, height, healthy_area, medium_area, unhealthy_area,
                                         predicted_yield, location, model_type, rice_variety, fertilizer_type)
            VALUES (:width, :height, :healthy, :medium, :unhealthy, :yield, :location, :model_type, :rice_variety, :fertilizer_type)
            RETURNING id
            """
            values = {
                "width": width,
                "height": height,
                "healthy": result["stats"]["healthy"],
                "medium": result["stats"]["medium"],
                "unhealthy": result["stats"]["unhealthy"],
                "yield": result["estimated_yield"],
                "location": location,
                "model_type": result["model_type"],
                "rice_variety": rice_variety,
                "fertilizer_type": fertilizer_type
            }
            record_id = await database.execute(query, values)
            result["record_id"] = record_id
        else:
            result["record_id"] = None
            result["db_warning"] = "Database not connected, record not saved"

        return JSONResponse(content=result)
    except Exception as e:
        print(f"Analyze endpoint error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

# (Other endpoints like /update_fertilizer, /save_actual_yield, /history remain unchanged)
# ---------- FRONTEND ----------
frontend_path = pathlib.Path("../frontend")
if frontend_path.exists() and frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    @app.get("/")
    async def read_root():
        return {"message": "API is running but frontend files not found"}

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"message": "Not found", "requested_path": full_path})

# ---------- STARTUP & SHUTDOWN ----------
@app.on_event("startup")
async def startup():
    print("Starting application...")
    db_connected = await init_db()
    global model
    try:
        model = joblib.load('yield_model.pkl')
        print("Loaded trained yield prediction model")
    except:
        model = None
        print("No trained model available, using linear model")
    if db_connected:
        await train_model()
    else:
        print("Database not connected, skipping model training")
    print("Application startup complete")

@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
