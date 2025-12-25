# Medical LLM Benchmarking Tool

A comprehensive tool for running medical question-answering across multiple large language models (both open-source and proprietary) to compare their performance on biomedical tasks.

## Overview

This tool evaluates medical LLMs by running a set of questions through various models and saving their responses for comparison. It supports both generative models and encoder models (for embeddings).

### System Prompt (Medical Expert Persona)

All generative models are grounded with a comprehensive system prompt that establishes the context as a renowned research scientist specializing in oncology and breast cancer research. This ensures:

- **Consistent Expert Persona**: Models respond as an experienced medical researcher with 20+ years of clinical experience
- **Evidence-Based Responses**: Emphasis on current medical evidence and clinical guidelines
- **Professional Medical Terminology**: Appropriate use of medical vocabulary while remaining accessible
- **Safety Considerations**: Clear acknowledgment that responses are informational and not personal medical advice
- **Research Context**: Background in both clinical practice and laboratory research

The system prompt grounds all responses in medical expertise while maintaining scientific rigor and patient safety considerations.

## Models Included

The tool is configured to support the following models. You can enable/disable specific models by commenting them out in `medical_model_runner.py`.

### Generative Medical Models (Text Generation)

1. **Google MedGemma 27B** (`google/medgemma-27b-text-it`)
   - 27B parameter text-only model
   - Instruction-tuned for medical reasoning

2. **PMC-LLaMA 13B** (`axiong/PMC_LLaMA_13B`)
   - 13B parameter model
   - Fine-tuned on 4.8M PubMed papers

3. **BioMedLM 2.7B** (`stanford-crfm/BioMedLM`)
   - 2.7B parameter GPT-style model
   - Trained on PubMed abstracts and full articles

### Proprietary Models (API-based)

1. **GPT-4o** (`gpt-4o`)
   - OpenAI's latest flagship model

2. **Gemini 3 Pro Preview** (`gemini-3-pro-preview`)
   - Google's latest experimental model

## Configuration

To select which models to run, edit `medical_model_runner.py`:

```python
# Uncomment models you want to run
MODELS_HF_GEN = [
    "google/medgemma-27b-text-it",
    # "axiong/PMC_LLaMA_13B", 
]

OPENAI_MODELS = ["gpt-4o"] 
GEMINI_MODELS = [] # ["gemini-3-pro-preview"]
```

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers openai google-generativeai bitsandbytes accelerate sentencepiece protobuf
```

### 2. Set Up API Keys

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-key"

# Google Gemini API Key
export GEMINI_API_KEY="your-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### 5. Create Questions File

Create a file named `questions.jsonl` with your medical questions in JSONL format:

```json
{"question": "What are the symptoms of type 2 diabetes?"}
{"question": "How does metformin work?"}
{"question": "What is the difference between systolic and diastolic blood pressure?"}
{"question": "Explain the mechanism of action of statins."}
{"question": "What are the risk factors for cardiovascular disease?"}
```

## Customizing the System Prompt

The system prompt establishes the expert persona and response guidelines. You can customize it in the code:

```python
SYSTEM_PROMPT = """You are a renowned research scientist and medical expert with extensive experience in oncology, particularly breast cancer research...
"""
```

### Alternative System Prompts

**General Medical Expert:**
```python
SYSTEM_PROMPT = """You are an experienced physician with broad medical knowledge across multiple specialties. Provide clear, evidence-based medical information suitable for both healthcare professionals and educated patients."""
```

**Specific Specialty (Cardiology):**
```python
SYSTEM_PROMPT = """You are a board-certified cardiologist with 15+ years of clinical experience. Specialize in cardiovascular disease, interventional cardiology, and heart failure management. Provide detailed, evidence-based cardiovascular medical information."""
```

**Patient Education Focus:**
```python
SYSTEM_PROMPT = """You are a medical educator specializing in patient communication. Explain complex medical concepts in clear, accessible language while maintaining scientific accuracy. Focus on helping patients understand their health conditions."""
```

**Research-Focused:**
```python
SYSTEM_PROMPT = """You are a medical research scientist analyzing clinical data and literature. Provide detailed scientific explanations with emphasis on research methodology, statistical analysis, and current evidence from peer-reviewed studies."""
```

## Usage

### Basic Usage

```bash
python medical_model_runner.py
```

### What Happens

1. The script reads all questions from `questions.jsonl`
2. For each model:
   - Creates an output folder (e.g., `google_medgemma-27b-text-it/`)
   - Runs each question through the model
   - Saves responses as numbered text files (`1.txt`, `2.txt`, etc.)

### Output Structure

```
project/
├── medical_model_runner.py
├── questions.jsonl
├── google_medgemma-27b-text-it/
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
├── axiong_PMC_LLaMA_13B/
│   ├── 1.txt
│   └── ...
├── stanford-crfm_BioMedLM/
│   └── ...
└── ...
```

## Model-Specific Notes

### HuggingFace Models

- **MedGemma 27B**: Requires accepting the Health AI Developer Foundation's terms on HuggingFace
- **PMC-LLaMA**: May show legacy tokenizer warnings (can be ignored)
- **BioMedLM**: Requires 11GB disk space for download

### Quantization

The script uses 4-bit quantization (`load_in_4bit=True`) for large models to reduce memory requirements:
- 27B model: ~15GB VRAM (vs ~54GB full precision)
- Trade-off: Slightly reduced quality for much better efficiency

To disable quantization (requires high VRAM):
```python
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    # model_kwargs={"load_in_4bit": True}  # Comment out this line
)
```

### Encoder Models

Encoder models return **embeddings** (vector representations) rather than text:
- Output is a string representation of the embedding vector
- Useful for similarity comparison, clustering, or downstream tasks
- Not directly human-readable

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size or use CPU for encoders
# Use smaller models (BioMedLM instead of MedGemma)
# Ensure no other processes are using GPU
```

### API Rate Limits

```python
# Add delay between API calls if needed
import time
time.sleep(1)  # Add after each API call
```

### Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/large/disk"

# Use HuggingFace CLI for manual download
huggingface-cli download google/medgemma-27b-text-it
```

## Performance Considerations

- **Generative models:** ~5-30 seconds per question (depending on model size)
- **Encoder models:** ~1-5 seconds per question
- **API models:** ~2-10 seconds per question (depends on network)
- **System prompt overhead:** Minimal (~50-100 tokens added per question)

For 100 questions across all 7 models: expect ~1-2 hours total runtime.

### Response Quality

The system prompt significantly improves response quality by:
- Establishing expert medical context
- Encouraging evidence-based reasoning
- Maintaining consistent professional tone
- Reducing hallucinations through role grounding
- Adding appropriate medical disclaimers

### Token Usage

With the system prompt:
- Average input: ~350 tokens (system prompt + question)
- Average output: ~300-512 tokens
- Total per question: ~650-850 tokens

## Model Comparison

After running, you can compare model outputs:

```python
import os
import json

def compare_responses(question_num):
    folders = [f for f in os.listdir() if os.path.isdir(f) and f != "__pycache__"]
    for folder in folders:
        file_path = os.path.join(folder, f"{question_num}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"\n{folder}:")
                print(f.read())

# Compare all models' responses to question 1
compare_responses(1)
```

## License

Check individual model licenses on HuggingFace:
- MedGemma: Health AI Developer Foundation terms
- PMC-LLaMA: OpenRAIL license
- BioMedLM: Apache 2.0
- BioLinkBERT: Apache 2.0
- BiomedBERT: MIT License
- SapBERT: MIT License

## Citations

If using these models in research, please cite:

```bibtex
@article{medgemma2025,
  title={MedGemma Technical Report},
  author={Sellergren et al.},
  journal={arXiv preprint arXiv:2507.05201},
  year={2025}
}

@article{pmcllama2023,
  title={PMC-LLaMA: Towards Building Open-source Language Models for Medicine},
  author={Wu, Chaoyi et al.},
  year={2023}
}

@article{biomedlm2024,
  title={BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text},
  author={Bolton et al.},
  journal={arXiv:2403.18421},
  year={2024}
}
```

## Contributing

Feel free to:
- Add more medical models
- Improve error handling
- Add evaluation metrics
- Create visualization tools

## Support

For issues:
- HuggingFace models: Check model cards on huggingface.co
- OpenAI API: Check status.openai.com
- Gemini API: Check Google AI Studio documentation