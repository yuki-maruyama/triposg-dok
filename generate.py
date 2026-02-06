#!/usr/bin/env python3
"""TripoSG (2025 SOTA) Generator for Sakura DOK"""
import os
import sys
import traceback

artifact_dir = os.environ.get('SAKURA_ARTIFACT_DIR', '/opt/artifact')
os.makedirs(artifact_dir, exist_ok=True)
log_path = os.path.join(artifact_dir, 'run.log')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, 'w')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

def main():
    import requests
    from io import BytesIO
    from PIL import Image
    import torch
    
    image_url = os.environ.get('IMAGE_URL')
    if not image_url:
        print("ERROR: IMAGE_URL not set")
        sys.exit(1)
    
    print(f"Input: {image_url}")
    
    # Download image
    print("Downloading image...")
    headers = {'User-Agent': 'TripoSG-DOK/1.0'}
    response = requests.get(image_url, timeout=60, headers=headers)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Image: {image.size}")
    
    # Load pipeline
    print("Loading TripoSG pipeline...")
    from triposg import TripoSGPipeline
    pipeline = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG")
    pipeline.to("cuda")
    
    # Generate
    print("Generating 3D mesh...")
    with torch.no_grad():
        mesh = pipeline(image)
    
    # Save
    output_path = os.path.join(artifact_dir, 'output.glb')
    mesh.export(output_path)
    print(f"Saved: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        log_file.close()
