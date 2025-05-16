import os
from flows.flow1_autonomous import process_autonomous
from flows.flow2_targeted import process_targeted_fields
from flows.flow4_hybrid import process_hybrid
import json

def read_file_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read(), os.path.splitext(file_path)[-1].lower()

if __name__ == "__main__":
    input_path = "assets/sample_invoice.png"
    file_bytes, file_ext = read_file_bytes(input_path)

    print("\n[FLOW 1] Autonomous Field Mapping")
    df1 = process_autonomous(file_bytes, file_ext)
    df1.to_csv("output_flow1.csv", index=False)
    print("[✓] Flow 1 completed → output_flow1.csv")

    print("\n[FLOW 2] Targeted Field Extraction")
    fields = ["Invoice Number", "Date", "Total Amount"]
    df2 = process_targeted_fields(file_bytes, file_ext, json.dumps(fields))
    df2.to_csv("output_flow2.csv", index=False)
    print("[✓] Flow 2 completed → output_flow2.csv")

    print("\n[FLOW 3] NLP (Using Hybrid)")
    prompt = "Extract all items with price and quantity"
    process_hybrid(file_bytes, file_ext, fields=["Item", "Quantity", "Price"], output_csv="output_flow3.csv")
    print("[✓] Flow 3 completed → output_flow3.csv")

    print("\n[FLOW 4] Hybrid NLP + Field Merge")
    fields = ["Invoice Number", "Date"]
    prompt = "Include any penalty clauses with amounts"
    process_hybrid(file_bytes, file_ext, fields, output_csv="output_flow4.csv")
    print("[✓] Flow 4 completed → output_flow4.csv")
