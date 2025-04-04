from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
import random

# Import only the necessary functions
from utils import find_json_for_label, load_sequence_from_json, slugify, predict_label_from_json

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.post("/get-sign-sequence")
async def get_sign_sequence(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        # Process the text into a sequence of frames
        all_frames = []
        words = [slugify(word) for word in text.strip().split()]
        word_sequences = []

        for word in words:
            json_path = find_json_for_label(word)
            
            # Word exists in dataset
            if json_path:
                sequence = load_sequence_from_json(json_path)
                word_sequences.append({
                    "word": word,
                    "frames": sequence
                })
                continue
                
            # Fallback to letters
            if word.isalpha():
                letter_frames = []
                for char in word:
                    char_path = find_json_for_label(char)
                    if char_path:
                        sequence = load_sequence_from_json(char_path)
                        letter_frames.extend(sequence)
                
                if letter_frames:
                    word_sequences.append({
                        "word": word,
                        "frames": letter_frames
                    })
                continue
                
            # Predict closest known sign
            samples = []
            for lbl in os.listdir("sign_dataset"):
                path = find_json_for_label(lbl)
                if path:
                    samples.append(path)
                    
            if samples:
                json_path = random.choice(samples)
                predicted_label = predict_label_from_json(json_path)
                json_path = find_json_for_label(predicted_label)
                
                if json_path:
                    sequence = load_sequence_from_json(json_path)
                    word_sequences.append({
                        "word": word,
                        "predicted_as": predicted_label,
                        "frames": sequence
                    })
        
        return {
            "success": True,
            "text": text,
            "sequences": word_sequences
        }
    
    except Exception as e:
        print(f"Error generating sequence data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Keep the health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}