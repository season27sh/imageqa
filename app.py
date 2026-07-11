from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer_image(data: RequestData):

    try:
        image_data = data.image_base64

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                image,
                f"""
Answer the following question using only the image.

Question:
{data.question}

Rules:
- Return ONLY the answer.
- Do not explain.
- If numeric, return only the number.
- No currency symbols.
- No units unless asked.
"""
            ]
        )

        answer = response.text.strip()

        answer = answer.replace("```", "")
        answer = answer.replace("`", "")
        answer = answer.replace("₹", "")
        answer = answer.replace("Rs.", "")
        answer = answer.replace("Rs", "")
        answer = answer.strip()
        
        print("=" * 50)
        print("QUESTION:", data.question)
        print("ANSWER:", answer)
        print("=" * 50)
        
        return {"answer": answer}

    except Exception as e:
        return {"answer": f"ERROR: {str(e)}"}
