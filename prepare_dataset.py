import os
import shutil
from pathlib import Path

def prepare_data():
    base_source = Path(r"c:\ML_Projects\Main\optic-nerve-cnn-master\optic-nerve-cnn-master\data")
    target_base = Path(r"c:\ML_Projects\model\data")
    
    # Create target directories
    (target_base / "normal").mkdir(parents=True, exist_ok=True)
    (target_base / "glaucoma").mkdir(parents=True, exist_ok=True)
    
    # Mapping
    sources = {
        "normal": [
            base_source / "HRF" / "Healthy",
            base_source / "RIM-ONE v1" / "Normal"
        ],
        "glaucoma": [
            base_source / "HRF" / "Glaucomatous",
            base_source / "RIM-ONE v1" / "Deep",
            base_source / "RIM-ONE v1" / "Moderate",
            base_source / "RIM-ONE v1" / "Early"
        ]
    }
    
    for category, paths in sources.items():
        count = 0
        for path in paths:
            if not path.exists():
                print(f"Warning: Path {path} does not exist. Skipping.")
                continue
            for img_file in path.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Copy to target with unique name
                    target_name = f"{category}_{count}{img_file.suffix}"
                    shutil.copy(img_file, target_base / category / target_name)
                    count += 1
        print(f"Prepared {count} images for {category}")

    # Create a sample image for frontend demo
    sample_src = next((target_base / "normal").glob("*"), None)
    if sample_src:
        frontend_public = Path(r"c:\ML_Projects\frontend\public")
        frontend_public.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_src, frontend_public / "sample_eye.jpg")
        print("Updated frontend/public/sample_eye.jpg")

if __name__ == "__main__":
    prepare_data()
