#!/usr/bin/env python3
"""TripoSG Generator for Sakura DOK - uses inference script"""
import os
import sys
import subprocess
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
    
    input_path = '/tmp/input.jpg'
    with open(input_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved input image: {len(response.content)} bytes")
    
    # Run TripoSG inference
    output_path = os.path.join(artifact_dir, 'output.glb')
    print("Running TripoSG inference...")
    
    cmd = [
        'python3', '-m', 'scripts.inference_triposg',
        '--image-input', input_path,
        '--output-path', output_path
    ]
    
    result = subprocess.run(
        cmd,
        cwd='/app/triposg',
        capture_output=True,
        text=True
    )
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Inference failed with code {result.returncode}")
        sys.exit(1)
    
    if os.path.exists(output_path):
        print(f"Success! Output: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print("ERROR: Output file not created")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        log_file.close()
