import os
import shutil
import glob

# Root to downloaded dataset
source_root = "/home/adam/.cache/kagglehub/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes/versions/1"
# New folder to organized data
dest_root = "/home/adam/MRI_project/yolo_dataset" 
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

folders_to_scan = {
    "train": os.path.join(source_root, "Train"),
    "val": os.path.join(source_root, "Val")
}

# New folder structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(dest_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dest_root, "labels", split), exist_ok=True)

print(f"Created folder structure in: {dest_root}")

# Go through the data and copy files
for split, source_folder in folders_to_scan.items():
    print(f"Processing data {split}")
    
    # Destination folders for split
    dest_image_dir = os.path.join(dest_root, "images", split)
    dest_label_dir = os.path.join(dest_root, "labels", split)

    for class_name in classes:
        source_image_dir = os.path.join(source_folder, class_name, "images")
        source_label_dir = os.path.join(source_folder, class_name, "labels")
        
        # Find all types of images
        image_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
             image_files.extend(glob.glob(os.path.join(source_image_dir, ext)))
        
        print(f"Founded {len(image_files)} images for class '{class_name}'")

        # Copy each image and label
        for img_path in image_files:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            label_file = base_name + ".txt"
            source_label_path = os.path.join(source_label_dir, label_file)
            
            shutil.copy(img_path, os.path.join(dest_image_dir, file_name))
            
            if os.path.exists(source_label_path):
                shutil.copy(source_label_path, os.path.join(dest_label_dir, label_file))

print("\n Organizing ended")