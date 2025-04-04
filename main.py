from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil

# Import your text_to_animation function
from utils import text_to_animation

app = FastAPI()

# Allow requests from your frontend with better CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create videos directory if it doesn't exist
os.makedirs("generated_videos", exist_ok=True)

# Mount the videos directory
app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

@app.post("/generate-sign-video")
async def generate_sign_video(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join("generated_videos", filename)
        
        # Generate the video
        text_to_animation(text, output_path)
        
        # Ensure the file exists and is readable
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to generate video")
        
        # Set file permissions to 666 (read/write for all users)
        os.chmod(output_path, 0o666)
        
        # Return both relative and absolute URLs
        video_url = f"/videos/{filename}"
        absolute_url = f"http://localhost:8000/videos/{filename}"
        
        return {
            "video_url": video_url, 
            "absolute_url": absolute_url,
            "success": True
        }
    
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")

# Direct file serving endpoint as a backup method
@app.get("/direct-video/{filename}")
async def get_video(filename: str):
    video_path = os.path.join("generated_videos", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Set permissions before serving
    os.chmod(video_path, 0o666)
    
    return FileResponse(
        video_path, 
        media_type="video/mp4",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )

# Add a debug endpoint to check video directory
@app.get("/debug/fix-permissions")
async def fix_permissions():
    """Fix permissions on all video files"""
    video_dir = "generated_videos"
    fixed_files = []
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
    
    for filename in os.listdir(video_dir):
        file_path = os.path.join(video_dir, filename)
        
        if os.path.isfile(file_path):
            # Set permissions to 666
            os.chmod(file_path, 0o666)
            
            fixed_files.append({
                "name": filename,
                "new_permissions": "666"
            })
    
    return {"status": "Permissions fixed", "files": fixed_files}
    

@app.get("/debug/generate-test-video")
async def generate_test_video():
    """Generate a simple test video to check if video creation works"""
    import cv2
    import numpy as np
    import os
    import time
    
    # Create test directory
    os.makedirs("generated_videos", exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time())
    filename = f"test_video_{timestamp}.mp4"
    output_path = os.path.join("generated_videos", filename)
    
    # Video properties
    width, height = 640, 480
    fps = 20.0
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    success = False
    file_info = {"exists": False, "size": 0, "permissions": ""}
    
    try:
        # Generate 30 test frames
        for i in range(30):
            # Create a blank frame
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Draw a moving circle
            center = (width // 2 + int(100 * np.sin(i / 5)), height // 2)
            cv2.circle(frame, center, 50, (0, 0, 255), -1)
            
            # Add frame number
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Write the frame
            out.write(frame)
        
        # Release the writer
        out.release()
        
        # Check file
        if os.path.exists(output_path):
            file_info["exists"] = True
            file_info["size"] = os.path.getsize(output_path)
            file_info["permissions"] = oct(os.stat(output_path).st_mode)[-3:]
            
            # Set permissions
            os.chmod(output_path, 0o644)
            
            success = True
    except Exception as e:
        return {"success": False, "error": str(e), "file_info": file_info}
    
    return {
        "success": success,
        "file_info": file_info,
        "video_url": f"/videos/{filename}"
    }

@app.get("/debug/list-videos")
async def list_videos():
    """List all videos in the generated_videos directory"""
    video_dir = "generated_videos"
    
    if not os.path.exists(video_dir):
        return {"error": f"Directory '{video_dir}' does not exist"}
    
    files = []
    for filename in os.listdir(video_dir):
        file_path = os.path.join(video_dir, filename)
        
        if os.path.isfile(file_path):
            files.append({
                "name": filename,
                "size": os.path.getsize(file_path),
                "permissions": oct(os.stat(file_path).st_mode)[-3:],
                "url": f"/videos/{filename}"
            })
    
    return {"files": files}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}

