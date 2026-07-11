from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from PIL import Image
import base64
import io
import os
import re
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


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/answer-image")
def answer_image(data: RequestData):

    try:
        image_data = data.image_base64.strip()

        # Remove data URL prefix if present
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        # Fix missing padding
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"""
You are an OCR and visual document question answering engine.

Carefully inspect the image and answer the user's question.

Question:
{data.question}

Rules:

1. Return ONLY the final answer.
2. Do NOT explain.
3. Do NOT include labels.
4. Do NOT use markdown.
5. Do NOT wrap in code blocks.
6. If the answer is numeric:
   - return digits only
   - keep decimal point if needed
   - remove commas
   - remove currency symbols
   - remove units
7. If the answer is text:
   - return only the requested text.
8. Read directly from the image.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image,
                prompt
            ]
        )

        answer = response.text.strip()

        # Cleanup
        answer = answer.replace("```", "")
        answer = answer.replace("`", "")
        answer = answer.replace("₹", "")
        answer = answer.replace("Rs.", "")
        answer = answer.replace("Rs", "")
        answer = answer.replace("$", "")
        answer = answer.replace(",", "")
        answer = answer.strip()

        # Remove common prefixes
        answer = re.sub(
            r"^(Answer:|The answer is:|The answer is)\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        print("=" * 60)
        print("QUESTION:", data.question)
        print("RAW ANSWER:", response.text)
        print("FINAL ANSWER:", answer)
        print("=" * 60)

        return {
            "answer": answer
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "answer": f"ERROR: {str(e)}"
        }
