# flow4_hybrid.py
import sys
import os
import pandas as pd
import json
from flow2_targeted import process_targeted_fields
from flow3_nlp import process_hybrid

def merge_dataframes(df1, df2):
    if df1 is None or df1.empty:
        return df2
    if df2 is None or df2.empty:
        return df1
    return pd.concat([df1, df2], axis=1).loc[:, ~pd.concat([df1, df2], axis=1).columns.duplicated()]

def run_flow4(input_file, fields, output_csv="final_output.csv"):
    file_ext = os.path.splitext(input_file)[-1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        print("[!] Unsupported file type")
        return

    with open(input_file, "rb") as f:
        file_bytes = f.read()

    print("[1] Running Flow 2 (targeted)...")
    fields_json = json.dumps(fields)
    df_targeted = process_targeted_fields(file_bytes, file_ext, fields_json)

    print("[2] Running Flow 3 (NLP)...")
    process_hybrid(file_bytes, file_ext, fields, "nlp_output.csv")
    df_nlp = pd.read_csv("nlp_output.csv")

    print("[3] Merging results...")
    df_merged = merge_dataframes(df_targeted, df_nlp)
    df_merged.to_csv(output_csv, index=False)
    print(f"[✓] Flow 4 complete. Saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python flow4_hybrid.py <input_file> <field1> <field2> ... <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_csv = sys.argv[-1]
    field_args = sys.argv[2:-1]
    
    run_flow4(input_path, field_args, output_csv)
