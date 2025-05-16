import pandas as pd
import re
import json
import cv2
import numpy as np
import sys
import os
from utils.ocr_utils import preprocess_image, extract_text_lines, convert_pdf_to_images

def match_fields(text_lines, target_fields):
    """
    Search each target field in the extracted text lines.
    If found, capture the first number-like value after the field.
    Otherwise, mark as 'NA'.
    """
    result = {}
    for field in target_fields:
        found = "NA"
        for line in text_lines:
            if re.search(re.escape(field), line, re.IGNORECASE):
                # Extract the first number in the line as value
                match = re.search(r'[\d\.,]+', line)
                found = match.group(0) if match else line.strip()
                break
        result[field] = found
    return result

def process_targeted_fields(file_bytes, file_ext, fields_json):
    """
    Process the input file (PDF or image) to extract text lines,
    then find the targeted fields in those lines.
    Returns a DataFrame with the results.
    """
    target_fields = json.loads(fields_json)
    lines = []

    if file_ext == ".pdf":
        images = convert_pdf_to_images(file_bytes)
        for img in images:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            preprocessed = preprocess_image(img_cv)
            lines += extract_text_lines(preprocessed)
    else:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        preprocessed = preprocess_image(img)
        lines = extract_text_lines(preprocessed)

    data = match_fields(lines, target_fields)
    return pd.DataFrame([data])

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python flow2_targeted.py <input_file> <field1> <field2> ... <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[-1]
    field_args = sys.argv[2:-1]  # All args between input and output are target fields

    file_ext = os.path.splitext(input_path)[-1].lower()

    with open(input_path, "rb") as f:
        file_bytes = f.read()

    fields_json = json.dumps(field_args)
    df_result = process_targeted_fields(file_bytes, file_ext, fields_json)
    df_result.to_csv(output_path, index=False)
    print(f"[✓] Saved output to {output_path}")
