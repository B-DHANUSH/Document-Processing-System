import pandas as pd
import pytesseract
import pdfplumber
from PIL import Image
from io import BytesIO
import sys
import os
import re

# 🔧 Optional: Set Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


def extract_ocr_lines(image):
    text = pytesseract.image_to_string(image)
    return [line.strip() for line in text.split("\n") if line.strip()]

def parse_item_line(line):
    """
    Parses a single OCR line to extract Qty, Item, and Price.
    """
    price_match = re.search(r'(\d+[\.,]\d{2})$', line)
    if price_match:
        price = price_match.group(1).replace(',', '.')
        text_before_price = line[:price_match.start()].strip()

        qty_match = re.match(r'(?:(\d+x|Qty[:\-]?\s*\d+|\d+)\s+)?(.+)', text_before_price)
        if qty_match:
            qty = qty_match.group(1) or ""
            item = qty_match.group(2).strip()
            return [qty, item, price]
    return None

def parse_lines(lines):
    structured = []
    for line in lines:
        parsed = parse_item_line(line)
        if parsed:
            structured.append(parsed + [""])  # Info column blank
        else:
            structured.append(["", "", "", line])  # Put line as 'Info'
    return structured

def process_autonomous(file_bytes, file_ext):
    structured_data = []

    if file_ext == ".pdf":
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                image = page.to_image(resolution=300).original
                lines = extract_ocr_lines(image)
                structured_data.extend(parse_lines(lines))
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        lines = extract_ocr_lines(image)
        structured_data.extend(parse_lines(lines))

    return pd.DataFrame(structured_data, columns=["Qty", "Item", "Price", "Info"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flow1_autonomous.py <input_file> [output_csv]")
        sys.exit(1)

    file_path = sys.argv[1]
    file_ext = os.path.splitext(file_path)[-1].lower()

    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        print("Supported formats: .png, .jpg, .jpeg, .pdf")
        sys.exit(1)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    df_result = process_autonomous(file_bytes, file_ext)

    if df_result.empty:
        print("[!] No data extracted.")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    df_result.to_csv(output_path, index=False)
    print(f"[✓] Saved structured output to {output_path}")
