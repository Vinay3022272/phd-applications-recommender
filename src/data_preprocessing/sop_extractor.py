import os
import json
import random
import pymupdf  # PyMuPDF
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONSTANTS (Update these paths as needed)
# ==========================================
INPUT_PATH = r"C:\Users\ps302\OneDrive\Desktop\Recommend\src\data\raw\phd\sop"
OUTPUT_PATH = r"C:\Users\ps302\OneDrive\Desktop\Recommend\src\data\processed\phd\sop\up_extract.json"

# ==========================================
# 2. SCHEMA & LLM SETUP
# ==========================================
class PhDScholarSOP(BaseModel):
    name: str = Field(
        default="UNK",
        description="The full name of the PhD applicant/scholar. If missing, keep it 'UNK'."
    )
    research_interests: List[str] = Field(
        default_factory=lambda: ["UNK"],
        description="List of primary research interests or topics mentioned in the SOP. If missing, keep it ['UNK']."
    )
    expertise: List[str] = Field(
        default_factory=lambda: ["UNK"],
        description="Comprehensive list of technical, research, laboratory, programming, and analytical skills/expertise. If missing, keep it ['UNK']."
    )
    department: str = Field(
        default="UNK",
        description="The academic department of the scholar. If missing, keep it 'UNK'."
    )
    university: str = Field(
        default="UNK",
        description="The target or current university/institution. If missing, keep it 'UNK'."
    )
    country: str = Field(
        default="UNK",
        description="The country of the scholar or university. If missing, keep it 'UNK'."
    )
    publications: List[str] = Field(
        default_factory=lambda: ["UNK"],
        description="List of published papers, preprints, or projects mentioned in the SOP. If missing, keep it ['UNK']."
    )
    publication_count: Union[int, str] = Field(
        default="UNK",
        description="Number of publications mentioned, or 'UNK' if missing."
    )
    citation_count: Union[int, str] = Field(
        default="UNK",
        description="Citation count mentioned, or 'UNK' if missing."
    )
    years_experience: Union[int, str] = Field(
        default="UNK",
        description="Years of research or work experience mentioned, or 'UNK' if missing."
    )

# Initialize the model
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.0,
    base_url="http://127.0.0.1:11434",
    format="json"
)

# Force the Ollama model to emit JSON matching the Pydantic schema
structured_llm = llm.with_structured_output(
    PhDScholarSOP,
    method="json_schema"
)

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert academic admissions reviewer. Extract structured details from the PhD candidate's Statement of Purpose (SOP).\n"
        "You must respond ONLY with a raw JSON object containing the fields specified in the schema. Do NOT wrap the JSON in markdown code blocks, do NOT write markdown tables, bullet points, or any explanatory text. Start directly with '{{' and end with '}}'.\n"
        "Fields to extract:\n"
        "- name: full name of scholar (or 'UNK')\n"
        "- research_interests: list of research interests/topics (or ['UNK'])\n"
        "- expertise: list of technical, analytical, research, and laboratory skills/expertise (or ['UNK'])\n"
        "- department: academic department (or 'UNK')\n"
        "- university: university or institution (or 'UNK')\n"
        "- country: country of scholar or institution (or 'UNK')\n"
        "- publications: list of publication titles/citations mentioned (or ['UNK'])\n"
        "- publication_count: number of publications mentioned (or 'UNK')\n"
        "- citation_count: total citations mentioned (or 'UNK')\n"
        "- years_experience: years of research or work experience mentioned (or 'UNK')\n\n"
        "If any field is missing or not mentioned in the SOP, use 'UNK' for string/number fields or ['UNK'] for list fields."
    ),
    ("human", "Statement of Purpose:\n\n{sop_text}")
])

# Create the runnable chain globally so it isn't recreated in the loop
extraction_chain = prompt | structured_llm


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        doc = pymupdf.open(pdf_path)
        full_text = [page.get_text() for page in doc]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Failed to read PDF {pdf_path}: {e}")
        return ""


# ==========================================
# 4. MAIN BATCH PROCESSING FUNCTION
# ==========================================
def process_sop_batches(input_dir: str, output_dir: str, batch_size: int = 10):
    """
    Scans the input directory for PDFs, extracts SOP data in batches using an LLM,
    and resumes from the last saved state without overwriting previous data.
    """
    # Handle direct file path vs directory path for output_dir
    if output_dir.lower().endswith(".json"):
        output_file_path = output_dir
        if os.path.exists(output_file_path) and os.path.isdir(output_file_path):
            try:
                sub_file = os.path.join(output_file_path, "extracted_sops.json")
                if os.path.exists(sub_file):
                    os.remove(sub_file)
                os.rmdir(output_file_path)
                print(f"Cleaned up legacy directory '{output_file_path}' to use it as a file.")
            except Exception as cleanup_err:
                print(f"Note: Could not clean up directory '{output_file_path}': {cleanup_err}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "extracted_sops.json")
    
    # --- RESUME LOGIC: Load existing data if it exists ---
    all_extracted_data = []
    processed_files = set()
    
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, "r", encoding="utf-8") as f:
                all_extracted_data = json.load(f)
                # Re-index existing records sequentially starting from 1 to avoid duplicates/legacy random IDs
                for idx, record in enumerate(all_extracted_data):
                    record["scholar_id"] = idx + 1
                # Create a set of filenames we have already successfully processed
                processed_files = {record.get("source_file") for record in all_extracted_data if "source_file" in record}
            print(f"Resuming... Found {len(processed_files)} previously processed files in the JSON.")
        except json.JSONDecodeError:
            print("Warning: Existing JSON file is corrupted or empty. Starting fresh but preserving the file.")
    
    # Get a list of all PDF files in the input directory
    all_pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    # Filter out the files that have already been processed
    pending_pdf_files = [
        os.path.join(input_dir, f) 
        for f in all_pdf_files 
        if f not in processed_files
    ]
    
    total_files = len(all_pdf_files)
    total_pending = len(pending_pdf_files)
    total_processed = len(processed_files)
    
    if total_pending == 0:
        print("All files in the directory have already been processed! Exiting.")
        return

    print(f"Found {total_files} total PDF(s). {total_pending} remaining to process.")
    
    # Process the remaining files in chunks of 'batch_size'
    for i in range(0, total_pending, batch_size):
        batch_files = pending_pdf_files[i : i + batch_size]
        batch_number = (i // batch_size) + 1
        
        print(f"\n--- Processing Batch {batch_number} ({len(batch_files)} files) ---")
        
        for idx_in_batch, pdf_path in enumerate(batch_files):
            file_name = os.path.basename(pdf_path)
            current_count = total_processed + i + idx_in_batch + 1
            print(f"[{current_count}/{total_files}] Extracting: {file_name}")
            
            # 1. Extract text from PDF
            sop_text = extract_text_from_pdf(pdf_path)
            if not sop_text.strip():
                print(f"   -> Warning: No text found in {file_name}. Skipping.")
                continue
            
            # 2. Run LLM Extraction
            try:
                extracted_data: PhDScholarSOP = extraction_chain.invoke({"sop_text": sop_text})
                
                # Convert Pydantic object to dict
                record = extracted_data.model_dump()
                scholar_name = record.get("name", "UNK").strip()
                
                # Check if scholar name already exists (case-insensitive, ignoring "UNK")
                if scholar_name != "UNK":
                    existing_names = {r.get("name", "").strip().lower() for r in all_extracted_data}
                    if scholar_name.lower() in existing_names:
                        print(f"   -> Skip: Scholar '{scholar_name}' already exists.")
                        continue
                
                # Append source filename and assign unique sequential scholar_id
                record["source_file"] = file_name
                record["scholar_id"] = len(all_extracted_data) + 1
                
                all_extracted_data.append(record)
                print(f"   -> Success: Extracted {scholar_name}")
                
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                if len(err_msg) > 120:
                    err_msg = err_msg[:120] + "..."
                print(f"   -> Error: {err_msg}")
        
        # 3. Save accumulated data after every batch (overwrites with the full updated list)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(all_extracted_data, f, indent=4)
            
        print(f"Batch {batch_number} complete. Progress saved to '{output_file_path}'.")

    print("\n==========================================")
    print(f"All batches processed! Total profiles extracted: {len(all_extracted_data)}")
    print(f"Final output saved to: {output_file_path}")
    print("==========================================")


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    process_sop_batches(
        input_dir=INPUT_PATH, 
        output_dir=OUTPUT_PATH, 
        batch_size=10
    )