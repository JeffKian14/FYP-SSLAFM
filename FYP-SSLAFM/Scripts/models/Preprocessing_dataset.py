import os
import cv2
import dlib
import numpy as np 

# ==========================================
# 1. SETUP & PATHS
# ==========================================
def setup():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: models -> Scripts -> FYP-SSLAFM
    root_folder = os.path.dirname(os.path.dirname(current_folder))
    print(f"Project Root found at: {root_folder}")
    return root_folder, current_folder

def folder_input(root_folder):
    dataset_name = input("Enter the name of dataset folder to preprocess (e.g., CASME2): ")
    input_path = os.path.join(root_folder, "Dataset", dataset_name)
    
    # OUTPUT FOLDER: No longer 'TESTING', this is the real deal
    output_path = os.path.join(root_folder, "Dataset", "models_Preprocess", f"{dataset_name}_preprocessed")

    if os.path.exists(input_path):
        return input_path, output_path
    else:
        print(f"Error: Folder not found at {input_path}")
        return None, None

# ==========================================
# 2. PROCESSING LOGIC (FULL SCALE)
# ==========================================
def process_dataset(input_path, output_path):
    # Initialize Dlib
    detector = dlib.get_frontal_face_detector()
    
    print(f"\nScanning ALL files in: {input_path}...")
    
    # --- STEP A: COLLECT ALL FILES FIRST ---
    all_image_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".jpg")):
                all_image_paths.append(os.path.join(root, file))
    
    total_images = len(all_image_paths)
    if total_images == 0:
        print("No images found! Check your folder structure.")
        return

    print(f"Found {total_images} images.")
    print(f"Saving to: {output_path}")
    print("Starting full processing... (This may take a while)\n")

    count = 0
    errors = 0

    # --- STEP B: PROCESS EVERYTHING ---
    for img_path in all_image_paths:
        try:
            # 1. Load Image (BGR Color)
            color_img = cv2.imread(img_path)
            if color_img is None:
                continue

            # 2. Noise Reduction (Bilateral Filter)
            # Preserves edges while smoothing skin
            clean_img = cv2.bilateralFilter(color_img, 9, 75, 75)

            # 3. Prepare for Dlib (Gray + 8-bit Safe)
            gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # 4. Detect Faces
            faces = detector(gray, 1)

            if len(faces) > 0:
                # Find largest face
                face = max(faces, key=lambda r: r.width() * r.height())
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Boundary Checks
                img_h, img_w = color_img.shape[:2]
                x, y = max(0, x), max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                # 5. Crop from Clean Color Image
                face_crop = clean_img[y:y+h, x:x+w] 
                
                if face_crop.size > 0:
                    # Resize to 224x224 (Standard for ResNet)
                    face_resized = cv2.resize(face_crop, (224, 224))
                    
                    # 6. Save Logic (Mirror Folder Structure)
                    rel_dir = os.path.relpath(os.path.dirname(img_path), input_path)
                    save_dir = os.path.join(output_path, rel_dir)
                    
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    filename = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(save_dir, filename), face_resized)
                    
                    count += 1
                    
                    # Progress Bar: Print status every 50 images
                    if count % 50 == 0:
                        print(f"Processed: {count}/{total_images} images...", end='\r')
                else:
                    errors += 1
            else:
                errors += 1
                # Optional: Uncomment below to see exactly which files fail
                # print(f"Skipped (No Face): {os.path.basename(img_path)}")
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(img_path)}: {e}")
            errors += 1

    print(f"\n\nDone! Successfully processed {count} images.")
    print(f"Skipped/Errors: {errors}")
    print(f"Output folder: {output_path}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    root_folder, current_folder = setup()
    input_dir, output_dir = folder_input(root_folder)

    if input_dir:
        process_dataset(input_dir, output_dir)
        