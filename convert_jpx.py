import os
import io
from PIL import Image

for fn in ['paper_fig_1_113.bin', 'paper_fig_2_214.bin', 'paper_fig_3_259.bin', 'paper_fig_4_294.bin']:
    with open(fn, 'rb') as f:
        data = f.read()
    try:
        img = Image.open(io.BytesIO(data))
        out_name = fn.replace('.bin', '.png')
        img.save(out_name)
        print(f"Successfully converted {fn} -> {out_name} (size: {img.size}, mode: {img.mode})")
    except Exception as e:
        print(f"Error converting {fn}: {e}")
