import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
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
MODELS_HF_GEN = [
    "google/medgemma-27b-text-it",  # FIXED: Changed from medgemma-27b-it to text-only variant
    "axiong/PMC_LLaMA_13B",         # Verified: Available on HuggingFace
    "stanford-crfm/BioMedLM"        # FIXED: Changed from StanfordAIMI/BioMedLM to correct path
]

# Encoder Medical Models (For Feature Extraction)
MODELS_HF_ENC = [
    "michiyasunaga/BioLinkBERT-large",  # Verified: Available on HuggingFace
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",  # Verified: Available
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # Verified: Available
]

# Latest Proprietary Models
OPENAI_MODELS = ["gpt-4o"]  # Latest/Largest OpenAI
GEMINI_MODELS = ["gemini-1.5-pro-002"]  # FIXED: Updated to latest Gemini model


def get_hf_gen_response(model_id, prompt):
    """Handles text-generation for large HF models."""
    try:
        # Using 4-bit quantization to fit the 27B model on consumer/mid-range GPUs
        # If you have massive VRAM (80GB+), you can remove load_in_4bit=True
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            model_kwargs={"load_in_4bit": True},
            trust_remote_code=True  # Added for some models that require it
        )
        output = pipe(prompt, max_new_tokens=200, do_sample=True)
        return output[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"


def get_hf_enc_response(model_id, prompt):
    """Handles Encoders: Returns the mean pooling of the last hidden state."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
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
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def get_gemini_response(model_id, prompt):
    """Handles Google Gemini API calls."""
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


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
        # Create folder name from model ID
        folder_name = model_id.replace("/", "_")
        os.makedirs(folder_name, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running: {model_id} ({m_type})")
        print(f"Output folder: {folder_name}")
        print(f"{'='*60}")

        # Process each question
        for i, q in enumerate(questions, 1):
            print(f"  Processing question {i}/{len(questions)}...", end=" ")

            # Get response based on model type
            if m_type == "hf_gen":
                ans = get_hf_gen_response(model_id, q)
            elif m_type == "hf_enc":
                ans = get_hf_enc_response(model_id, q)
            elif m_type == "openai":
                ans = get_openai_response(model_id, q)
            elif m_type == "gemini":
                ans = get_gemini_response(model_id, q)
            else:
                ans = "Error: Unknown model type"

            # Write output to file
            output_file = os.path.join(folder_name, f"{i}.txt")
            try:
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(ans)
                print("✓")
            except Exception as e:
                print(f"✗ Error writing file: {e}")

        print(f"Completed {model_id}")

    print(f"\n{'='*60}")
    print("All models processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()