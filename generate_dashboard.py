
import json
import os
import json
import os
import html
import markdown

def generate_dashboard():
    # 1. Load Questions
    questions = []
    if os.path.exists("questions.jsonl"):
        with open("questions.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line)["question"])

    # 2. Define Models and Paths
    models = [
        ("PMC-LLaMA 13B", "axiong_PMC_LLaMA_13B"),
        ("GPT-4o", "gpt-4o"),
        ("Gemini 3 Pro", "gemini-3-pro-preview"),
        ("MedGemma 27B", "google_medgemma-27b-text-it"),
        ("Perplexity", "perplexity"),
        ("OpenEvidence", "openevidence"),
    ]

    # 3. Build Data Structure
    # db_data = { question_index: { model_name: response_text, ... }, ... }
    db_data = {}
    
    for i, question_text in enumerate(questions, 1):
        db_data[i] = {
            "question": question_text,
            "responses": []
        }
        for model_name, folder in models:
            # Try .md first, then .txt
            text_content = "[No response file find]"
            
            # Paths to check
            paths = [f"{folder}/{i}.md", f"{folder}/{i}.txt"]
            
            for p in paths:
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            text_content = f.read()
                        break
                    except Exception as e:
                        text_content = f"Error reading file: {e}"
            
            # Convert Markdown to HTML
            # We use the markdown library to render formatting (bold, lists, etc)
            html_content = markdown.markdown(text_content)
            
            db_data[i]["responses"].append({
                "model": model_name,
                "text": html_content
            })

    # 4. Generate HTML
    # We embed the data directly into the HTML as a JSON object
    json_data = json.dumps(db_data)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Model Eval Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; margin: 0; padding: 20px; }}
        header {{ margin-bottom: 20px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        h1 {{ margin: 0 0 10px 0; font-size: 24px; color: #1d1d1f; }}
        
        /* Controls */
        .controls {{ display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }}
        select {{ padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #d2d2d7; min-width: 300px; max-width: 100%; }}
        button {{ padding: 10px 20px; background: #0071e3; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; transition: background 0.2s; }}
        button:hover {{ background: #0077ed; }}
        .q-display {{ margin-top: 15px; font-size: 18px; font-weight: 600; color: #1d1d1f; line-height: 1.4; }}

        /* Grid Layout */
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        
        .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #e5e5e5; display: flex; flex-direction: column; }}
        .card h3 {{ margin-top: 0; font-size: 16px; color: #86868b; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        .card-content {{ flex-grow: 1; font-size: 15px; line-height: 1.6; color: #333; overflow-x: auto; white-space: pre-wrap; margin-top: 10px; }}
        
        /* Model Specific Colors */
        .card[data-model*="GPT"] h3 {{ color: #10a37f; border-bottom-color: #10a37f; }}
        .card[data-model*="Gemini"] h3 {{ color: #4285f4; border-bottom-color: #4285f4; }}
        .card[data-model*="BioMed"] h3 {{ color: #ea4335; border-bottom-color: #ea4335; }}
        .card[data-model*="MedGemma"] h3 {{ color: #fbbc04; border-bottom-color: #fbbc04; }}
    </style>
</head>
<body>

<header>
    <h1>Medical Model Evaluation Dashboard</h1>
    <div class="controls">
        <label for="q-select">Select Question:</label>
        <select id="q-select" onchange="renderQuestion()">
            <!-- Options populated by JS -->
        </select>
        <button onclick="changeQ(-1)">← Prev</button>
        <button onclick="changeQ(1)">Next →</button>
    </div>
    <div class="q-display" id="question-text">
        <!-- Question text goes here -->
    </div>
</header>

<div class="grid" id="response-grid">
    <!-- Response cards go here -->
</div>

<script>
    // Embedded Data
    const DATA = {json_data};
    const Q_IDS = Object.keys(DATA).sort((a,b) => parseInt(a)-parseInt(b));
    let currentQIndex = 0;

    const selectEl = document.getElementById('q-select');
    const qTextEl = document.getElementById('question-text');
    const gridEl = document.getElementById('response-grid');

    function init() {{
        // Populate Dropdown
        Q_IDS.forEach((id, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            const text = DATA[id].question;
            opt.text = `Q${{id}}: ` + (text.length > 50 ? text.substring(0, 50) + '...' : text);
            selectEl.appendChild(opt);
        }});
        
        renderQuestion();
    }}

    function changeQ(delta) {{
        const newIndex = parseInt(selectEl.value) + delta;
        if (newIndex >= 0 && newIndex < Q_IDS.length) {{
            selectEl.value = newIndex;
            renderQuestion();
        }}
    }}

    function renderQuestion() {{
        currentQIndex = selectEl.value;
        const qId = Q_IDS[currentQIndex];
        const data = DATA[qId];

        // Update Header
        qTextEl.textContent = `Q${{qId}}: ${{data.question}}`;

        // Update Grid
        gridEl.innerHTML = '';
        
        data.responses.forEach(res => {{
            const card = document.createElement('div');
            card.className = 'card';
            card.setAttribute('data-model', res.model);
            
            // Convert simple markdown-like bolding if needed, but otherwise keep as text
            // The python script already replaced newlines with <br>
            
            card.innerHTML = `
                <h3>${{res.model}}</h3>
                <div class="card-content">${{res.text}}</div>
            `;
            gridEl.appendChild(card);
        }});
    }}

    init();
</script>

</body>
</html>
"""

    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print("Successfully generated 'dashboard.html'")

if __name__ == "__main__":
    generate_dashboard()
