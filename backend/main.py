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
import schedule
import asyncio
import traceback
import uvicorn
import os
from databases import Database
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import pathlib
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL configuration - use Render's environment variable
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:mamerto@localhost:5432/rice_yield')

# Fix for Render's PostgreSQL URL format
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# For Render PostgreSQL, add SSL requirement
if DATABASE_URL and ("render.com" in DATABASE_URL or "onrender.com" in DATABASE_URL):
    # Parse the connection URL to properly handle SSL
    parsed = urlparse(DATABASE_URL)
    
    # Extract query parameters
    query_params = parse_qs(parsed.query)
    
    # Add or update sslmode parameter
    query_params['sslmode'] = ['require']
    
    # Rebuild the URL with updated query
    new_query = urlencode(query_params, doseq=True)
    DATABASE_URL = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))

# Create database with SSL for production
if DATABASE_URL and ("render.com" in DATABASE_URL or "onrender.com" in DATABASE_URL):
    database = Database(DATABASE_URL, ssl=True)
else:
    database = Database(DATABASE_URL)

# Rice variety yield factors
RICE_VARIETY_FACTORS = {
    "Jasmine": 1.0,
    "Basmati": 0.95,
    "Arborio": 1.1,
    "Calrose": 1.05,
    "Japonica": 0.98,
    "Indica": 1.02
}

# Fertilizer effectiveness
FERTILIZER_EFFECTIVENESS = {
    "Urea": 0.9,
    "DAP": 1.0,
    "NPK": 1.15,
    "Organic": 0.85
}

# Determine the correct path for static files
current_dir = pathlib.Path(__file__).parent
print(f"Current directory: {current_dir}")

# Check multiple possible locations for frontend files
frontend_paths = [
    current_dir.parent / "frontend",  # ../frontend
    current_dir / "frontend",         # ./frontend
    pathlib.Path("/opt/render/project/src/frontend")  # Render's default path
]

served_static = False
for frontend_path in frontend_paths:
    if frontend_path.exists() and frontend_path.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        print(f"Serving static files from: {frontend_path}")
        served_static = True
        break

if not served_static:
    print("Frontend directory not found in any expected location.")
    # List available directories for debugging
    print("Available directories:")
    for path in frontend_paths:
        print(f"  {path}: {'Exists' if path.exists() else 'Does not exist'}")
    
    # Create a simple root endpoint for debugging
    @app.get("/")
    async def read_root():
        return {
            "message": "API is running but frontend files not found",
            "current_dir": str(current_dir),
            "frontend_paths": [str(p) for p in frontend_paths]
        }

# Initialize DB with retry logic
async def init_db():
    max_retries = 5
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            await database.connect()
            # Create table if not exists
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
            print(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries exceeded. Database connection failed.")
                # Don't crash the app if database connection fails
                # The app can still work for image processing, just not save to DB
                return False

# Model training
async def train_model():
    try:
        print("Starting yield prediction model training...")
        query = "SELECT * FROM historical_data WHERE actual_yield IS NOT NULL"
        rows = await database.fetch_all(query)
        df = pd.DataFrame([dict(row) for row in rows])

        if df.empty or len(df) < 10:
            print("Insufficient data for training (min 10 records)")
            return False

        # Ensure columns exist
        if 'rice_variety' not in df.columns:
            df['rice_variety'] = 'Jasmine'
        if 'fertilizer_type' not in df.columns:
            df['fertilizer_type'] = 'Urea'

        df['variety_factor'] = df['rice_variety'].map(RICE_VARIETY_FACTORS)
        df['fertilizer_factor'] = df['fertilizer_type'].map(FERTILIZER_EFFECTIVENESS)
        df.fillna({'variety_factor': 1.0, 'fertilizer_factor': 1.0}, inplace=True)

        X = df[['healthy_area', 'medium_area', 'unhealthy_area', 
                'width', 'height', 'variety_factor', 'fertilizer_factor']]
        y = df['actual_yield']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'yield_model.pkl')
        print(f"Model trained with {len(df)} records")
        return True
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        return False

# Load model if exists
model = None

# Scheduler task
async def scheduler_task():
    schedule.every().day.at("02:00").do(lambda: asyncio.run(train_model()))
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)

# Image validation
def is_valid_image_type(content_type: str, filename: str) -> bool:
    valid_types = ["image/jpeg", "image/jpg", "image/png"]
    valid_extensions = [".jpg", ".jpeg", ".png"]
    if content_type.lower() in valid_types:
        return True
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        return True
    return False

def is_rice_field(image_np: np.ndarray) -> bool:
    try:
        hsv = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.count_nonzero(green_mask) / (image_np.shape[0] * image_np.shape[1]) * 100
        return green_percentage > 30
    except Exception as e:
        print(f"Rice field detection error: {str(e)}")
        traceback.print_exc()
        return False

def process_image(image_data: bytes, field_width_m: float, field_height_m: float, 
                  rice_variety: str, fertilizer_type: str):
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
        m2_per_pixel = (field_width_m * field_height_m) / total_pixels

        healthy_area = np.sum(mask_healthy) * m2_per_pixel
        medium_area = np.sum(mask_medium) * m2_per_pixel
        unhealthy_area = np.sum(mask_unhealthy) * m2_per_pixel

        variety_factor = RICE_VARIETY_FACTORS.get(rice_variety, 1.0)
        fertilizer_factor = FERTILIZER_EFFECTIVENESS.get(fertilizer_type, 1.0)

        model_type = "linear"
        global model
        if model:
            input_data = [[
                healthy_area, medium_area, unhealthy_area,
                field_width_m, field_height_m,
                variety_factor, fertilizer_factor
            ]]
            yield_kg = model.predict(input_data)[0]
            model_type = "ML"
        else:
            base_yield = (healthy_area * 0.8) + (medium_area * 0.4) + (unhealthy_area * 0.1)
            yield_kg = base_yield * variety_factor * fertilizer_factor

        output_pil = Image.fromarray(out_img.astype(np.uint8))
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        processed_image_b64 = base64.b64encode(buffered.getvalue()).decode()

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
        print(f"Image processing error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Image processing failed: {str(e)}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Try to execute a simple query
        await database.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

# Pydantic model for update fertilizer
class UpdateFertilizerRequest(BaseModel):
    record_id: int
    fertilizer_type: str
    rice_variety: str
    healthy_area: float
    medium_area: float
    unhealthy_area: float
    width: float
    height: float

# Pydantic model for saving actual yield
class SaveActualYieldRequest(BaseModel):
    actualYield: float
    record_id: int
    rice_variety: str
    fertilizer_type: str

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
            return JSONResponse(
                status_code=400,
                content={"error": "Only JPG, JPEG, and PNG files are allowed"}
            )
        contents = await image.read()
        result = process_image(contents, width, height, rice_variety, fertilizer_type)
        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})

        # Save to DB
        query = """
            INSERT INTO historical_data 
            (width, height, healthy_area, medium_area, unhealthy_area, 
             predicted_yield, location, model_type, rice_variety, fertilizer_type)
            VALUES (:width, :height, :healthy, :medium, :unhealthy, 
                    :yield, :location, :model_type, :rice_variety, :fertilizer_type)
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
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/update_fertilizer")
async def update_fertilizer(request: UpdateFertilizerRequest):
    try:
        variety_factor = RICE_VARIETY_FACTORS.get(request.rice_variety, 1.0)
        fertilizer_factor = FERTILIZER_EFFECTIVENESS.get(request.fertilizer_type, 1.0)

        model_type = "linear"
        global model
        if model:
            input_data = [[
                request.healthy_area, request.medium_area, request.unhealthy_area, 
                request.width, request.height,
                variety_factor, fertilizer_factor
            ]]
            yield_kg = model.predict(input_data)[0]
            model_type = "ML"
        else:
            base_yield = (request.healthy_area * 0.8) + (request.medium_area * 0.4) + (request.unhealthy_area * 0.1)
            yield_kg = base_yield * variety_factor * fertilizer_factor

        query = """
            UPDATE historical_data 
            SET fertilizer_type = :fertilizer_type, 
                predicted_yield = :predicted_yield,
                model_type = :model_type
            WHERE id = :record_id
        """
        values = {
            "fertilizer_type": request.fertilizer_type,
            "predicted_yield": yield_kg,
            "model_type": model_type,
            "record_id": request.record_id
        }
        await database.execute(query, values)
        return JSONResponse(content={
            "status": "success",
            "new_yield": yield_kg,
            "new_fertilizer": request.fertilizer_type,
            "model_type": model_type
        })
    except Exception as e:
        print(f"Error updating fertilizer: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/save_actual_yield")
async def save_actual_yield(request: SaveActualYieldRequest):
    try:
        query = """
            UPDATE historical_data 
            SET actual_yield = :actual_yield
            WHERE id = :record_id
        """
        values = {
            "actual_yield": request.actualYield, 
            "record极速": request.record_id
        }
        
        result = await database.execute(query, values)
        
        if result == 0:
            return JSONResponse(
                status_code=404, 
                content={"message": f"Record with ID {request.record_id} not found"}
            )
            
        print(f"Updated record {request.record_id} with actual yield: {request.actualYield}")

        # Retrain model with new data
        if await train_model():
            global model
            try:
                model = joblib.load('yield_model.pkl')
                print("Reloaded trained model after update")
            except:
                model = None
                print("Failed to reload model after update")

        return {"status": "success", "message": f"Updated actual yield to {request.actualYield} for record {request.record_id}"}
    except Exception as e:
        print(f"Error saving actual yield: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/history")
async def get_history(limit: int = Query(50, gt=0, le=100)):
    try:
        query = """
            SELECT id, timestamp, location, predicted_yield, actual_yield, 
                   model_type, rice_variety, fertilizer_type
            FROM historical_data 
            ORDER by timestamp DESC
            LIMIT :limit
        """
        rows = await database.fetch_all(query, {"limit": limit})
        history = []
        for row in rows:
            history.append({
                "id": row["id"],
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "location": row["location"],
                "predicted_yield": row["predicted_yield"],
                "actual_yield": row["actual_yield"],
                "model_type": row["model_type"],
                "rice_variety": row["rice_variety"],
                "fertilizer_type": row["fertilizer_type"]
            })
        return history
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

# Catch-all route to serve frontend for SPA routing
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Check if the requested file exists in any frontend directory
    for frontend_path in frontend_paths:
        if frontend_path.exists():
            file_path = frontend_path / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
    
    # If no file found, serve index.html for SPA routing
    for frontend_path in frontend_paths:
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    
    return JSONResponse(
        status_code=404,
        content={"message": "Not found", "requested_path": full_path}
    )

@app.on_event("startup")
async def startup():
    print("Starting application...")
    print(f"Current directory: {pathlib.Path.cwd()}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")
    
    # List files for debugging
    print("Current directory contents:")
    for item in pathlib.Path('.').iterdir():
        print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Try to connect to database but don't crash if it fails
    db_connected = await init_db()
    
    global model
    try:
        model = joblib.load('yield_model.pkl')
        print("Loaded trained yield prediction model")
    except:
        model = None
        print("No trained model available, using linear model")
    
    # Only start scheduler if database is connected
    if db_connected:
        await train_model()
        asyncio.create_task(scheduler_task())
    else:
        print("Database not connected, skipping model training and scheduler")
    
    print("Application startup complete")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)