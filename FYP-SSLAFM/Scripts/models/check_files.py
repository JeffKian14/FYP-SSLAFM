import os

# Path to one specific micro-expression folder
test_path = r"C:\Users\jeffk\FYP\FYP-SSLAFM\Dataset\CAS(ME)^2\15\anger1_1"

print(f"Checking: {test_path}")
if os.path.exists(test_path):
    files = sorted([f for f in os.listdir(test_path) if f.endswith('.jpg')])
    print(f"Total files: {len(files)}")
    print(f"First 3 files: {files[:3]}")
else:
    print("Folder path not found. Please verify the path.")