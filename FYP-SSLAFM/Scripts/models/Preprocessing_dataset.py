import os
import cv2
import dlib
import numpy as np 


def setup():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.dirname(os.path.dirname(current_folder))
    print(f"Project Root found at: {root_folder}")
    return root_folder, current_folder

def folder_input(root_folder):
    dataset_name = input("Enter the name of dataset folder to preprocess (CASME2): ")
    input_path = os.path.join(root_folder, "Dataset", dataset_name)
    output_path = os.path.join(root_folder, "Dataset", "models_Preprocess", f"{dataset_name}_preprocessed")

    if os.path.exists(input_path):
        return input_path, output_path
    else:
        print(f"Error: Folder not found at {input_path}")
        return None, None

def process_dataset(input_path, output_path):
    detector = dlib.get_frontal_face_detector()
    
    print(f"\nScanning ALL files in: {input_path}...")
    
    all_image = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".jpg")):
                all_image.append(os.path.join(root, file))
    
    total_images = len(all_image)
    if total_images == 0:
        print("No images found! Check your folder structure.")
        return None

    print(f"Found {total_images} images.")
    print(f"Saving to: {output_path}")
    print("Starting full processing...\n")

    count = 0
    errors = 0

    for image_path in all_image:
        try:
            color_img = cv2.imread(image_path)

            # skip if image not loaded
            if color_img is None:
                continue

            # Note: it can be deleted if needed, it helps smooth the edge
            clean_img = cv2.bilateralFilter(color_img, 9, 75, 75)

            # Dlib prefers grayscale images
            gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            faces = detector(gray, 1)

            if len(faces) > 0:
                face = max(faces, key=lambda r: r.width() * r.height())
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                img_h, img_w = color_img.shape[:2]
                x, y = max(0, x), max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                face_crop = clean_img[y:y+h, x:x+w] 
                
                if face_crop.size > 0:
                    # Resize to 224x224
                    face_resized = cv2.resize(face_crop, (224, 224))
                    
                    # Save to same file, then group by folders
                    relative_dir = os.path.relpath(os.path.dirname(image_path), input_path)
                    save_dir = os.path.join(output_path, relative_dir)
                    
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    filename = os.path.basename(image_path)
                    cv2.imwrite(os.path.join(save_dir, filename), face_resized)
                    
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"Processed: {count}/{total_images} images...", end='\r')
                else:
                    errors += 1
            else:
                errors += 1
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(image_path)}: {e}")
            errors += 1

    print(f"\n\nSuccessfully processed {count} images.")
    print(f"Errors: {errors} images could not be processed.")
    print(f"Output folder: {output_path}")


if __name__ == "__main__":
    root_folder, current_folder = setup()
    input_dir, output_dir = folder_input(root_folder)

    if input_dir:
        process_dataset(input_dir, output_dir)

        # solve unverified breakpoint
        input("\nProcessing complete. Press Enter to exit.")