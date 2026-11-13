"""
Create a contact sheet showing sample progression through training.
"""

from PIL import Image
import os

# Load sample grids
samples = [
    ('Step 0 (Initial)', 'outputs/samples_step_000000.png'),
    ('Step 500', 'outputs/samples_step_000500.png'),
    ('Step 1000', 'outputs/samples_step_001000.png'),
    ('Step 1500', 'outputs/samples_step_001500.png'),
    ('Step 2000 (Final)', 'outputs/samples_final.png'),
]

# Create contact sheet
images = []
labels = []

for label, path in samples:
    if os.path.exists(path):
        img = Image.open(path)
        images.append(img)
        labels.append(label)
        print(f"Loaded {label}")
    else:
        print(f"Warning: {path} not found")

if images:
    # Calculate grid dimensions
    n_cols = 2
    n_rows = (len(images) + 1) // 2

    # Get dimensions (all should be same size)
    img_width, img_height = images[0].size

    # Create contact sheet
    contact_width = n_cols * img_width
    contact_height = n_rows * img_height
    contact = Image.new('RGB', (contact_width, contact_height), 'white')

    # Paste images
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        x = col * img_width
        y = row * img_height
        contact.paste(img, (x, y))

    # Save
    contact.save('outputs/sample_progression_contact_sheet.png')
    print(f"\nContact sheet saved to outputs/sample_progression_contact_sheet.png")
else:
    print("No images loaded")
