#!/usr/bin/env python3
"""TripoSR Generator for Sakura DOK"""
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
    import shutil
    
    image_url = os.environ.get('IMAGE_URL')
    if not image_url:
        print("ERROR: IMAGE_URL not set")
        return False
    
    print(f"Input: {image_url}")
    
    # Download image
    print("Downloading image...")
    try:
        headers = {'User-Agent': 'TripoSR-DOK/1.0'}
        response = requests.get(image_url, timeout=60, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"ERROR: Failed to download image: {e}")
        return False
    
    input_path = '/tmp/input.png'
    with open(input_path, 'wb') as f:
        f.write(response.content)
    print(f"Saved input image: {len(response.content)} bytes")
    
    # Run TripoSR
    print("Running TripoSR...")
    
    cmd = [
        'python3', 'run.py',
        input_path,
        '--output-dir', artifact_dir,
        '--model-save-format', 'glb',
        '--pretrained-model-name-or-path', '/app/models/TripoSR'
    ]
    
    result = subprocess.run(
        cmd,
        cwd='/app/triposr',
        capture_output=True,
        text=True
    )
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"WARNING: Inference exited with code {result.returncode}")
        # Don't return False here - check for output first
    
    # Find output file (TripoSR outputs to 0/mesh.glb)
    mesh_path = os.path.join(artifact_dir, '0', 'mesh.glb')
    if os.path.exists(mesh_path):
        final_path = os.path.join(artifact_dir, 'output.glb')
        shutil.copy(mesh_path, final_path)
        print(f"Success! Output: {final_path}")
        print(f"File size: {os.path.getsize(final_path)} bytes")
        return True
    
    # Fallback: search for any GLB
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            if f.endswith('.glb'):
                src = os.path.join(root, f)
                final_path = os.path.join(artifact_dir, 'output.glb')
                shutil.copy(src, final_path)
                print(f"Success! Output: {final_path}")
                print(f"File size: {os.path.getsize(final_path)} bytes")
                return True
    
    print("ERROR: No GLB output found")
    return False

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("=== Task completed successfully ===")
        else:
            print("=== Task completed with errors (check log) ===")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        print("=== Task completed with exception ===")
    finally:
        log_file.close()
    # Always exit 0 for DOK
    sys.exit(0)
