
from pypdf import PdfReader
import re

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

reader = PdfReader("OpenEvidence.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

norm_text = normalize_text(full_text)

with open("pdf_debug.txt", "w") as f:
    f.write(norm_text)
