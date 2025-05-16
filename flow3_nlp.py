import sys
import os
import json
import re
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from phi.llm.google import Gemini
from phi.llm.message import Message
from utils.ocr_utils import preprocess_image, extract_text_lines, convert_pdf_to_images

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("[!] GOOGLE_API_KEY not found in .env")
    sys.exit(1)

ocr_agent = Gemini(api_key=api_key)

def generate_prompt(prompt, text):
    return f"{prompt}\n\nDocument OCR Text:\n{text}"

def process_nlp_prompt(file_bytes, file_ext, prompt, output_csv="output_flow3.csv"):
    lines = []
    if file_ext == ".pdf":
        images = convert_pdf_to_images(file_bytes)
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            pre = preprocess_image(img_cv)
            lines += extract_text_lines(pre)
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        pre = preprocess_image(img)
        lines = extract_text_lines(pre)

    document_text = "\n".join(lines)

    try:
        prompt_input = generate_prompt(prompt, document_text)
        messages = [Message(role="user", parts=[prompt_input])]
        response = ocr_agent.invoke(messages)

        cleaned = re.sub(r'^```(?:json)?|```$', '', response.text.strip(), flags=re.MULTILINE).strip()
        inferred_fields = json.loads(cleaned)

        if isinstance(inferred_fields, dict):
            inferred_fields = [inferred_fields]

        df = pd.DataFrame(inferred_fields)
        df.to_csv(output_csv, index=False)
        print(f"[✓] NLP output saved to {output_csv}")
    except Exception as e:
        print(f"[!] Gemini NLP Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python flow3_nlp.py <file_path> <prompt_text> <output_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    prompt = sys.argv[2]
    output_csv = sys.argv[3]

    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[-1].lower()
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    process_nlp_prompt(file_bytes, ext, prompt, output_csv)
