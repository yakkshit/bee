import zlib
import re

with open('toaf369.pdf', 'rb') as f:
    data = f.read()

streams = re.findall(rb'stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)

full_text = []
for i, s in enumerate(streams):
    try:
        d = zlib.decompress(s)
        if b'BT' in d and b'ET' in d:
            literals = re.findall(rb'\(((?:[^()\\]|\\.)*)\)', d)
            text_parts = []
            for lit in literals:
                s_lit = lit.decode('latin1', errors='ignore')
                text_parts.append(s_lit)
            full_text.append(' '.join(text_parts))
    except Exception as e:
        pass

combined = '\n--- PAGE/SECTION ---\n'.join(full_text)
with open('extracted_paper_clean.txt', 'w', encoding='utf-8') as f:
    f.write(combined)

print('Saved extracted text. Total chars:', len(combined))

# Find lines with keywords
for line in combined.split('\n'):
    line_s = line.strip()
    if any(k in line_s.lower() for k in ['fig.', 'figure', 'kaplan', 'survival', 'lt50', 'half-life', 'weibull', 'cox', 'probit', 'data analysis', 'statistical analysis', 'materials and methods']):
        print('MATCH:', line_s[:160])
