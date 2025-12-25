import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, BitsAndBytesConfig
from openai import OpenAI
import google.generativeai as genai

# --- Configuration: Using Largest Available Variants ---
INPUT_FILE = "questions.jsonl"

# System Prompt for Medical Context
SYSTEM_PROMPT = """You are a renowned research scientist and medical expert with extensive experience in oncology, particularly breast cancer research. You have:

- Over 20 years of clinical and research experience in oncology
- Published extensively in peer-reviewed medical journals
- Expertise in cancer biology, treatment protocols, and patient care
- Deep knowledge of current medical literature and clinical guidelines
- Experience with both clinical practice and laboratory research

Provide accurate, evidence-based medical information. When answering:
- Base responses on current medical evidence and established clinical guidelines
- Cite relevant research findings when applicable
- Acknowledge areas of ongoing research or uncertainty
- Use clear, professional medical terminology while remaining accessible
- Consider both clinical efficacy and patient safety

Remember: Your responses should be informative and scientifically accurate, but always note that individual medical decisions should be made in consultation with a qualified healthcare provider."""

# Generative Medical Models
# "google/medgemma-27b-text-it" Changed from medgemma-27b-it to text-only variant
MODELS_HF_GEN = [
    "google/medgemma-27b-text-it"     # Changed from medgemma-27b-it to text-only variant
    # "axiong/PMC_LLaMA_13B",         # Verified: Available on HuggingFace
    # "stanford-crfm/BioMedLM"        # FIXED: Changed from StanfordAIMI/BioMedLM to correct path
]

# Encoder Medical Models (For Feature Extraction)
MODELS_HF_ENC = []
# MODELS_HF_ENC = [
#     "michiyasunaga/BioLinkBERT-large",  # Verified: Available on HuggingFace
#     "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",  # Verified: Available
#     "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Verified: Available
# ]

# Latest Proprietary Models
OPENAI_MODELS = [] # "gpt-4o"]  # Latest/Largest OpenAI
GEMINI_MODELS = [] # ["gemini-3-pro-preview"]  # FIXED: Updated to latest Gemini model


def load_hf_gen_model(model_id):
    """Loads and returns the text-generation pipeline."""
    try:
        # Configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        return pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            model_kwargs={"quantization_config": bnb_config},
            trust_remote_code=True,
            use_fast=False  # Use slow tokenizer to avoid SentencePiece conversion errors
        )
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

def generate_hf_gen_response(pipe, prompt):
    """Generates text using the loaded pipeline."""
    try:
        # 1. Format prompt for Base Models (which expect continuation)
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        # 2. Check context length to avoid CUDA asserts (BioMedLM 1024 limit)
        model_config = pipe.model.config
        max_length = getattr(model_config, "max_position_embeddings", 2048)
        
        # Approximate token count (char count / 4 is a rough heuristic, but safer to use tokenizer)
        input_tokens = len(pipe.tokenizer.encode(formatted_prompt))
        
        # Calculate safe max_new_tokens
        # Reserve some buffer and ensure we don't exceed model limit
        max_new_tokens = 512 # Reduced default for safety and quality
        if input_tokens + max_new_tokens > max_length:
            max_new_tokens = max(100, max_length - input_tokens - 10)
            print(f"  [Info] Reduced max_new_tokens to {max_new_tokens} to fit context window ({max_length})")

        output = pipe(
            formatted_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.1,
            repetition_penalty=1.2, # Added to stop loops
            pad_token_id=pipe.tokenizer.eos_token_id, # Ensure padding is set
            return_full_text=False # Return ONLY the answer
        )
        return output[0]['generated_text'].strip()
    except Exception as e:
        return f"Error: {str(e)}"


def load_hf_enc_model(model_id):
    """Loads tokenizer and model for encoders."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

def generate_hf_enc_response(loaded_data, prompt):
    """Generates embedding using loaded encoder components."""
    try:
        tokenizer, model, device = loaded_data
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Return string representation of the embedding vector
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
        return str(embedding)
    except Exception as e:
        return f"Error: {str(e)}"


def get_openai_response(model_id, prompt):
    """Handles OpenAI API calls."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def get_gemini_response(model_id, prompt):
    """Handles Google Gemini API calls."""
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_id)
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"



def check_and_clean_disk_space(required_free_space_gb=30, exclude_model_id=None):
    """
    Checks if there is enough free disk space.
    If not, deletes the least recently accessed models from the Hugging Face cache
    until the required space is available.
    """
    try:
        from huggingface_hub import scan_cache_dir, constants
        import shutil
    except ImportError:
        print("Warning: huggingface_hub not installed. Skipping disk space check.")
        return

    # Determine cache directory
    cache_dir = constants.HF_HOME
    if not os.path.exists(cache_dir):
        # If cache dir doesn't exist, check space on the parent
        cache_dir = os.path.dirname(cache_dir)
        if not os.path.exists(cache_dir):
             cache_dir = os.getcwd()

    try:
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)
        
        if free_gb >= required_free_space_gb:
            return

        print(f"\\n[Disk Space Management] Low disk space ({free_gb:.2f} GB). Required: {required_free_space_gb} GB.")
        print("[Disk Space Management] Cleaning up old models...")

        hf_cache_info = scan_cache_dir()
        repos = list(hf_cache_info.repos)
        # Sort by last accessed (oldest first)
        repos.sort(key=lambda r: r.last_accessed)

        for repo in repos:
            if free_gb >= required_free_space_gb:
                break
            
            # Skip if it is the model we are currently processing
            if exclude_model_id and repo.repo_id == exclude_model_id:
                print(f"[Disk Space Management] Skipping {repo.repo_id} (currently in use/requested)")
                continue

            # Calculate size
            repo_size_gb = repo.size_on_disk / (1024**3)
            print(f"[Disk Space Management] Deleting {repo.repo_id} (Size: {repo_size_gb:.2f} GB)...")
            
            # Delete all revisions
            hashes = [rev.commit_hash for rev in repo.revisions]
            strategy = hf_cache_info.delete_revisions(*hashes)
            strategy.execute()
            
            # Recheck space
            total, used, free = shutil.disk_usage(cache_dir)
            free_gb = free / (1024**3)
            print(f"[Disk Space Management] New free space: {free_gb:.2f} GB")

        if free_gb < required_free_space_gb:
            print(f"[Disk Space Management] Warning: Could not reclaim enough space. Free: {free_gb:.2f} GB")
        else:
            print("[Disk Space Management] Cleanup complete.")
            
    except Exception as e:
        print(f"[Disk Space Management] Error: {e}")


def main():
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found!")
        print(f"Please create a file named '{INPUT_FILE}' with questions in JSONL format.")
        print('Example format: {"question": "What is diabetes?"}')
        return

    # Load questions from JSONL file
    try:
        with open(INPUT_FILE, 'r') as f:
            questions = [json.loads(line)['question'] for line in f]
        print(f"Loaded {len(questions)} questions from {INPUT_FILE}")
    except Exception as e:
        print(f"Error loading questions: {e}")
        return

    # Combined model list: (id, type)
    all_models = (
        [(m, "hf_gen") for m in MODELS_HF_GEN] +
        [(m, "hf_enc") for m in MODELS_HF_ENC] +
        [(m, "openai") for m in OPENAI_MODELS] +
        [(m, "gemini") for m in GEMINI_MODELS]
    )

    print(f"\nProcessing {len(all_models)} models:")
    for model_id, m_type in all_models:
        print(f"  - {model_id} ({m_type})")

    # Process each model
    for model_id, m_type in all_models:
        # Ensure sufficient disk space for HF models
        if m_type in ["hf_gen", "hf_enc"]:
             check_and_clean_disk_space(exclude_model_id=model_id)

        # Create folder name from model ID
        folder_name = model_id.replace("/", "_")
        abs_folder_path = os.path.abspath(folder_name)
        os.makedirs(abs_folder_path, exist_ok=True)
        print(f"Created/Verified folder: {abs_folder_path}")

        print(f"\n{'='*60}")
        print(f"Running: {model_id} ({m_type})")
        print(f"Output folder: {folder_name}")
        print(f"{'='*60}")

        # Prepare model context (load once)
        model_context = None
        if m_type == "hf_gen":
            print(f"Loading model {model_id}...")
            model_context = load_hf_gen_model(model_id)
            if model_context is None:
                print(f"Skipping {model_id} due to load failure.")
                continue
        elif m_type == "hf_enc":
            print(f"Loading model {model_id}...")
            model_context = load_hf_enc_model(model_id)
            if model_context is None:
                print(f"Skipping {model_id} due to load failure.")
                continue

        # Process each question
        for i, q in enumerate(questions, 1):
            print(f"  Processing question {i}/{len(questions)}...", end=" ")

            # Get response based on model type
            if m_type == "hf_gen":
                ans = generate_hf_gen_response(model_context, q)
            elif m_type == "hf_enc":
                ans = generate_hf_enc_response(model_context, q)
            elif m_type == "openai":
                ans = get_openai_response(model_id, q)
            elif m_type == "gemini":
                ans = get_gemini_response(model_id, q)
            else:
                ans = "Error: Unknown model type"

            # Write output to file
            output_file = os.path.join(abs_folder_path, f"{i}.txt")
            if not os.path.exists(abs_folder_path):
                 print(f"WARNING: Folder {abs_folder_path} mysteriously disappeared. Re-creating.")
                 os.makedirs(abs_folder_path, exist_ok=True)
            
            try:
                print(f"Writing to {output_file}...")
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(ans)
                print("✓")
            except Exception as e:
                print(f"✗ Error writing file: {e}")

        print(f"Completed {model_id}")
        
        # Cleanup memory for HF models - disabled to prevent CUDA errors
        # if model_context is not None:
        #     del model_context
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All models processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()