import json
import os
import time
import pandas as pd
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc="Progress", **kwargs):
            self.iterable = list(iterable)
            self.desc = desc
            self.total = len(self.iterable)
            self.postfix = ""
        def __iter__(self):
            for idx, item in enumerate(self.iterable):
                percent = 100 * (idx + 1) / self.total
                bar_len = 30
                filled = int(bar_len * (idx + 1) // self.total)
                bar = '█' * filled + '-' * (bar_len - filled)
                postfix_str = f" | {self.postfix}" if self.postfix else ""
                print(f"\r{self.desc}: |{bar}| {percent:.1f}% ({idx+1}/{self.total}){postfix_str}", end="", flush=True)
                yield item
            print()
        def set_postfix(self, postfix_dict):
            self.postfix = ", ".join(f"{k}: {v}" for k, v in postfix_dict.items())

# Configure client to point to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required string, but ignored by local Ollama
)

# Replace with your exact model tag in Ollama (e.g., "deepseek-r1:latest", "gpt-oss:120b-cloud", etc.)
MODEL_NAME = "gpt-oss:120b-cloud"

CATEGORIES = {
    "Engineering & Technology": [
        "AI & Autonomous Systems",
        "Geospatial Modeling & Remote Sensing",
        "Computational Geotechnics & Fluid Dynamics",
        "Sustainable Materials"
    ],
    "Social Sciences": [
        "Computational Social Science",
        "Behavioral Economics & Mechanism Design",
        "Public Policy & Urban Analytics"
    ],
    "Medical & Health Sciences": [
        "Precision Oncology & Genomic Medicine",
        "Epidemiological Machine Learning",
        "Translational Immunology"
    ],
    "Arts & Humanities": [
        "Digital Humanities & NLP",
        "Philosophy of AI & Ethics",
        "Computational Musicology"
    ],
    "Physical Sciences": [
        "Quantum Condensed Matter",
        "High-Energy Astrophysics",
        "Computational Geophysics"
    ],
    "Chemical Sciences": [
        "Heterogeneous Catalysis",
        "Polymer Informatics",
        "Electrochemical Energy Storage"
    ],
    "Agricultural Sciences": [
        "Precision Agronomy",
        "Soil Microbiology & Carbon Sequestration",
        "Climate-Resilient Crop Genomics"
    ],
    "Biological Sciences": [
        "Synthetic Biology",
        "Structural Bioinformatics",
        "Microbial Systems Biology"
    ]
}

BATCH_SIZE = 2
TARGET_PER_CATEGORY = 250  # 250 * 8 = 2000 total samples

SYSTEM_PROMPT = """You are an academic admissions evaluator. Generate authentic, publication-grade PhD candidate profiles with Statements of Purpose (SOPs).
Each profile must follow the formal structure:
- Header & Research Hook
- 1. Research Interests (Central research questions + 2 core pillars)
- 2. Why PhD at Target University (Faculty and lab fit)
- 3. Research Experience (Thematic breakdown, technical rigor, in-text citations [1, 2])
- 4. Career Goals & Mentorship
- References (Complete bibliography of cited papers)

The SOP must be structured as a list of plain text paragraphs (no markdown formatting like hashes '#', bolding '**', lists, or bullet points).
Return ONLY a valid JSON array."""

all_records = []
if os.path.exists("phd_academic_sops_2000.json"):
    try:
        with open("phd_academic_sops_2000.json", "r", encoding="utf-8") as f:
            all_records = json.load(f)
        print(f"Loaded {len(all_records)} existing records from phd_academic_sops_2000.json")
        
        # Ensure all existing JSON records conform to the ID scheme starting at 150
        updated = False
        for idx, record in enumerate(all_records):
            new_id = 150 + idx
            if record.get("scholar_id") != new_id:
                record["scholar_id"] = new_id
                record["source_file"] = f"generated_sop_{new_id}.pdf"
                updated = True
        if updated:
            with open("phd_academic_sops_2000.json", "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=4, ensure_ascii=False)
            print("Updated scholar_id indexing in phd_academic_sops_2000.json")
    except Exception as e:
        print(f"Could not load/verify existing records from JSON: {e}")
elif os.path.exists("phd_academic_sops_2000.csv"):
    try:
        df_existing = pd.read_csv("phd_academic_sops_2000.csv")
        # Keep non-empty rows and convert to list of dicts
        all_records = df_existing.dropna(subset=["category"]).to_dict(orient="records")
        print(f"Loaded {len(all_records)} existing records from legacy phd_academic_sops_2000.csv")
        
        # Parse list fields correctly and migrate legacy records
        import ast
        for idx, record in enumerate(all_records):
            # Rename legacy keys if present
            if "applicant_name" in record:
                record["name"] = record.pop("applicant_name")
            if "target_university" in record:
                record["university"] = record.pop("target_university")
                
            # Assign scholar_id starting from 150
            record["scholar_id"] = 150 + idx
            record["source_file"] = f"generated_sop_{150 + idx}.pdf"
            
            # Parse list fields
            for list_field in ["research_interests", "expertise", "publications", "sop_paragraphs"]:
                if list_field in record:
                    val = record[list_field]
                    if isinstance(val, str):
                        try:
                            record[list_field] = json.loads(val)
                        except Exception:
                            try:
                                record[list_field] = ast.literal_eval(val)
                            except Exception:
                                if list_field == "sop_paragraphs":
                                    record[list_field] = [p.strip() for p in val.split("\n\n") if p.strip()]
                                else:
                                    record[list_field] = [x.strip() for x in val.split(",") if x.strip()]
                else:
                    if list_field == "sop_paragraphs":
                        old_sop = record.get("full_sop_markdown", "")
                        if pd.isna(old_sop):
                            old_sop = ""
                        paragraphs = [p.strip() for p in str(old_sop).split("\n\n") if p.strip()]
                        cleaned_paragraphs = []
                        for p in paragraphs:
                            p_clean = p.replace("**", "").replace("*", "").lstrip("#").strip()
                            if p_clean:
                                cleaned_paragraphs.append(p_clean)
                        record["sop_paragraphs"] = cleaned_paragraphs
                    else:
                        record[list_field] = ["UNK"]
            
            record["sop_paragraphs_json"] = json.dumps(record.get("sop_paragraphs", []))
            
            # Fill other default keys
            if "department" not in record or pd.isna(record.get("department")):
                record["department"] = "UNK"
            if "country" not in record or pd.isna(record.get("country")):
                record["country"] = "UNK"
            if "publication_count" not in record or pd.isna(record.get("publication_count")):
                record["publication_count"] = len(record.get("publications", []))
            if "citation_count" not in record or pd.isna(record.get("citation_count")):
                record["citation_count"] = "UNK"
            if "years_experience" not in record or pd.isna(record.get("years_experience")):
                record["years_experience"] = "UNK"
            
            # Recreate full_sop_markdown and word_count
            cleaned_paragraphs = record.get("sop_paragraphs", [])
            record["full_sop_markdown"] = "\n\n".join(cleaned_paragraphs)
            sop_text = " ".join(cleaned_paragraphs)
            record["word_count"] = len(sop_text.split())

        # Save to JSON immediately to transition away from legacy CSV
        with open("phd_academic_sops_2000.json", "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=4, ensure_ascii=False)
        print("Successfully migrated legacy CSV records to phd_academic_sops_2000.json")
    except Exception as e:
        print(f"Could not load/migrate legacy CSV records: {e}")

for category, subdomains in CATEGORIES.items():
    batches = TARGET_PER_CATEGORY // BATCH_SIZE
    
    # Calculate how many records we already have for this category
    existing_for_cat = [r for r in all_records if r.get("category") == category]
    already_done = len(existing_for_cat)
    
    if already_done >= TARGET_PER_CATEGORY:
        print(f"\n================ Category: {category} is already fully generated ({already_done} samples). Skipping. ================")
        continue
        
    start_batch = already_done // BATCH_SIZE
    print(f"\n================ Starting Category: {category} (resuming from batch {start_batch + 1}/{batches}) ================")

    pbar = tqdm(range(start_batch, batches), desc=f"Generating {category}")
    for b in pbar:
        user_prompt = f"""Generate a batch of {BATCH_SIZE} distinct PhD candidate profiles and SOPs for:
Category: {category}
Sub-domains: {', '.join(subdomains)}

Output strictly valid JSON. For each candidate profile, generate a JSON object with these exact keys:
- 'name': Full name of the applicant.
- 'research_interests': List of 5-8 primary research interests or topics.
- 'expertise': List of 8-15 technical, research, laboratory, programming, and analytical skills/expertise.
- 'department': Academic department (e.g. Computer Science, Bioinformatics, Informatics).
- 'university': Target university.
- 'country': Target university's country (e.g. United States, United Kingdom).
- 'publications': List of 2-5 publication titles, preprints, or projects mentioned in their SOP.
- 'publication_count': Number of publications in the list (as an integer).
- 'citation_count': Total citations mentioned (as a string or integer, e.g., "12" or "UNK").
- 'years_experience': Years of research/work experience (as a string or integer, e.g., "3").
- 'sop_paragraphs': List of strings representing the paragraphs of the Statement of Purpose in plain text (without markdown formatting like '#', '**', or lists)."""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.75,
                # Enforce JSON output format supported by Ollama
                response_format={"type": "json_object"}
            )

            raw_output = response.choices[0].message.content.strip()

            # Clean any leftover markdown blocks
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()

            data = json.loads(raw_output)
            
            # Handle root wrapper if model returns {"sops": [...]} or raw [...]
            if isinstance(data, dict):
                data = next(iter(data.values()))
            new_batch_records = []
            for item in data:
                # Set missing category/sub_domain metadata keys if not present
                if "category" not in item:
                    item["category"] = category
                if "sub_domain" not in item:
                    item["sub_domain"] = subdomains[0] if subdomains else category
                
                # Ensure all required profile keys exist and match correct schema names
                if "name" not in item and "applicant_name" in item:
                    item["name"] = item.pop("applicant_name")
                elif "name" not in item:
                    item["name"] = "UNK"
                
                if "university" not in item and "target_university" in item:
                    item["university"] = item.pop("target_university")
                elif "university" not in item:
                    item["university"] = "UNK"
                    
                for list_field in ["research_interests", "expertise", "publications"]:
                    if list_field not in item:
                        item[list_field] = ["UNK"]
                    elif isinstance(item[list_field], str):
                        try:
                            item[list_field] = json.loads(item[list_field])
                        except Exception:
                            item[list_field] = [x.strip() for x in item[list_field].split(",") if x.strip()]
                            
                for str_field in ["department", "country", "publication_count", "citation_count", "years_experience"]:
                    if str_field not in item:
                        item[str_field] = "UNK"
                
                # Retrieve paragraphs list
                sop_paragraphs = item.get("sop_paragraphs", [])
                if isinstance(sop_paragraphs, str):
                    sop_paragraphs = [p.strip() for p in sop_paragraphs.split("\n\n") if p.strip()]
                
                # Clean up any markdown characters from LLM output
                cleaned_paragraphs = []
                for p in sop_paragraphs:
                    p_clean = p.replace("**", "").replace("*", "").lstrip("#").strip()
                    if p_clean:
                        cleaned_paragraphs.append(p_clean)
                
                item["sop_paragraphs"] = cleaned_paragraphs
                item["sop_paragraphs_json"] = json.dumps(cleaned_paragraphs)
                
                # Maintain full_sop_markdown (as a plain text double-newline joined string) for backwards compatibility
                item["full_sop_markdown"] = "\n\n".join(cleaned_paragraphs)
                
                # Combine paragraphs to get word count
                sop_text = " ".join(cleaned_paragraphs)
                item["word_count"] = len(sop_text.split())
                
                new_batch_records.append(item)

            if new_batch_records:
                # Load the latest records from the JSON file to prevent replacing/overwriting data
                current_records = []
                if os.path.exists("phd_academic_sops_2000.json"):
                    try:
                        with open("phd_academic_sops_2000.json", "r", encoding="utf-8") as f:
                            current_records = json.load(f)
                    except Exception as load_err:
                        current_records = list(all_records)
                
                # Assign sequential scholar_id based on loaded file record count
                for item in new_batch_records:
                    scholar_id = 150 + len(current_records)
                    item["scholar_id"] = scholar_id
                    item["source_file"] = f"generated_sop_{scholar_id}.pdf"
                    current_records.append(item)
                    all_records.append(item) # Keep loop tracker in sync
                
                # Write back to JSON file safely
                try:
                    with open("phd_academic_sops_2000.json", "w", encoding="utf-8") as f:
                        json.dump(current_records, f, indent=4, ensure_ascii=False)
                except Exception as json_err:
                    msg = f"Failed to save JSON: {json_err}"
                    if hasattr(pbar, "write"):
                        pbar.write(msg)
                    else:
                        print(msg)

            pbar.set_postfix({"total": len(all_records)})

        except Exception as err:
            msg = f"[{category}] Error on batch {b + 1}: {err}"
            if hasattr(pbar, "write"):
                pbar.write(msg)
            else:
                print(f"\n{msg}")
            
            # Stop execution immediately if we hit weekly usage/rate limits to allow clean resumes
            err_str = str(err).lower()
            if "429" in err_str or "too many requests" in err_str or "usage limit" in err_str or "limit reached" in err_str:
                if hasattr(pbar, "close"):
                    pbar.close()
                print("\n[CRITICAL] Weekly usage limit or rate limit reached. Exiting script to prevent skipping batches.")
                import sys
                sys.exit(1)
                
            time.sleep(2)

print(f"\nDataset generation completed/paused: {len(all_records)} samples saved to phd_academic_sops_2000.json.")