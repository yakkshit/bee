import zlib
import re
import os

with open('toaf369.pdf', 'rb') as f:
    data = f.read()

# find image objects
img_objs = re.findall(rb'(\d+)\s+0\s+obj\s*<<([^>]*/Subtype\s*/Image[^>]*)>>\s*stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL)
print(f"Found {len(img_objs)} images")

for i, (obj_id, header, stream) in enumerate(img_objs):
    header_str = header.decode('latin1', errors='ignore')
    print(f"\nImage {i+1} (obj {obj_id.decode()}): {header_str}")
    
    # Check filter
    if '/DCTDecode' in header_str:
        ext = 'jpg'
        out_data = stream
    elif '/FlateDecode' in header_str:
        ext = 'png'
        try:
            decomp = zlib.decompress(stream)
            # check width, height, color space
            w_m = re.search(r'/Width\s+(\d+)', header_str)
            h_m = re.search(r'/Height\s+(\d+)', header_str)
            w = int(w_m.group(1)) if w_m else None
            h = int(h_m.group(1)) if h_m else None
            print(f"  Decompressed bytes: {len(decomp)}, Width: {w}, Height: {h}")
            
            # Use PIL to save
            from PIL import Image
            if '/DeviceRGB' in header_str and w and h:
                img = Image.frombytes('RGB', (w, h), decomp)
                img.save(f"paper_fig_{i+1}_{obj_id.decode()}.png")
                continue
            elif '/DeviceGray' in header_str and w and h:
                img = Image.frombytes('L', (w, h), decomp)
                img.save(f"paper_fig_{i+1}_{obj_id.decode()}.png")
                continue
        except Exception as e:
            print("  Decompress/PIL error:", e)
        out_data = stream
    else:
        ext = 'bin'
        out_data = stream
        
    fn = f"paper_fig_{i+1}_{obj_id.decode()}.{ext}"
    with open(fn, 'wb') as f:
        f.write(out_data)
    print(f"  Saved {fn}")
