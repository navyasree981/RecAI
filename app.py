# app.py - Hugging Face Spaces entry point
# This is the required entry point for Hugging Face Spaces

from main import app

# Hugging Face Spaces will automatically run this
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)