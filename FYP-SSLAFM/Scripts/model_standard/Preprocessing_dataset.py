'Python 3.10 NumPy 1.26.4 dlib 19.24'
# pip install numpy==1.26.4
# pip install opencv-python
# pip install cmake
# pip install dlib
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
    dataset_name = input("Enter the name of dataset folder to preprocess (e.g., CASME2): ")
    input_path = os.path.join(root_folder, "Dataset", dataset_name)
    output_path = os.path.join(root_folder, "Dataset", "models_Preprocess", f"{dataset_name}_preprocessed")

    if os.path.exists(input_path):
        return input_path, output_path
    else:
        print(f"Error: Folder not found at {input_path}")
        return None, None

def process_dataset(input_path, output_path):
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    
    landmark_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(landmark_path):
        print(f"Error: {landmark_path} not found! Please download it.")
        return
    
    predictor = dlib.shape_predictor(landmark_path)
    
    print(f"\nScanning ALL files in: {input_path}...")
    
    all_images = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(os.path.join(root, file))
    
    total_images = len(all_images)
    if total_images == 0:
        print("No images found! Check your folder structure.")
        return None

    print(f"Found {total_images} images.")
    print(f"Saving to: {output_path}")
    print("Starting landmark-based processing...\n")

    count = 0
    errors = 0

    for image_path in all_images:
        try:
            color_img = cv2.imread(image_path)
            if color_img is None:
                continue

            # Pre-filtering for noise reduction
            clean_img = cv2.bilateralFilter(color_img, 9, 75, 75)
            gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            
            # Ensure the array is in the correct format for dlib
            gray = np.ascontiguousarray(gray, dtype=np.uint8)

            faces = detector(gray, 1)

            if len(faces) > 0:
                # Select the largest face detected
                face_rect = max(faces, key=lambda r: r.width() * r.height())
                
                # Get 68 landmarks 
                shape = predictor(gray, face_rect)
                
                # Convert landmarks to a numpy array of (x, y) coordinates
                coords = np.array([[p.x, p.y] for p in shape.parts()])

                # --- NEW FOCUSED CROP LOGIC ---
                # Get the exact extreme edges of the facial features
                min_x = np.min(coords[:, 0]) # Leftmost jaw point
                max_x = np.max(coords[:, 0]) # Rightmost jaw point
                min_y = np.min(coords[:, 1]) # Highest eyebrow point
                max_y = np.max(coords[:, 1]) # Lowest chin point

                # Calculate the true center of the actual features
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2

                # Find the largest dimension to ensure a perfect square crop 
                # (A square crop prevents the face from stretching when resized to 224x224)
                face_w = max_x - min_x
                face_h = max_y - min_y
                max_dim = max(face_w, face_h)

                # Set half_size to exactly half the max dimension, plus a tiny 5% padding.
                # Adjust 1.05 to 1.10 if it is slightly too tight.
                half_size = int((max_dim / 2) * 1.05) 
                # ------------------------------

                img_h, img_w = color_img.shape[:2]
                
                # Define boundaries with clipping to image size
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img_w, center_x + half_size)
                y2 = min(img_h, center_y + half_size)
                
                # Crop and resize
                face_crop = clean_img[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    
                    # Maintain folder structure in the output path
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
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(image_path)}: {e}")
            errors += 1

    print(f"\n\nSuccessfully processed {count} images.")
    print(f"Errors: {errors} images could not be processed.")

if __name__ == "__main__":
    root_folder, current_folder = setup()
    input_dir, output_dir = folder_input(root_folder)

    if input_dir:
        process_dataset(input_dir, output_dir)
        input("\nProcessing complete. Press Enter to exit.")