import zlib
import re
import os

with open('toaf369.pdf', 'rb') as f:
    data = f.read()

# Extract all images from PDF streams
# Find XObject with /Subtype /Image
img_pattern = re.compile(rb'<<.*?/Subtype\s*/Image.*?>>\s*stream[\r\n]+(.*?)[\r\n]+endstream', re.DOTALL)
matches = img_pattern.findall(data)
print(f"Found {len(matches)} image streams")

# Let's also extract text around Fig
with open('extracted_paper_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Find figure captions
for m in re.finditer(r'(Fig(?:ure)?\.\s*\d+[^.\n]*\..*?)(?=\n\n|\n[A-Z][a-z]+|\Z)', text, re.DOTALL):
    print("--- FIGURE CAPTION ---")
    print(m.group(0)[:500])
    print()

# Print section headers and statistical methods
for m in re.finditer(r'(Statistical analysis|Data analysis|Bioassay|Oral toxicity|Mortality).*?(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL):
    print("--- METHOD SECTION ---")
    print(m.group(0)[:600])
    print()
