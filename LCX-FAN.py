#@title 5、检测目录并保存张量：
import matplotlib.pyplot as plt
import face_alignment
from skimage import io
import numpy as np
import os
import torch # Import torch to save landmarks as tensors

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

# Directory containing images
image_directory = '/content/Deep3DFaceRecon_pytorch/ME'
# Directory to save landmarks
landmarks_directory = '/content/Deep3DFaceRecon_pytorch/ME/68lms'

# Create the landmarks directory if it doesn't exist
os.makedirs(landmarks_directory, exist_ok=True)

# Get list of image files
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]

for img_path in image_files:
    input_image = io.imread(img_path)
    preds = fa.get_landmarks(input_image)

    if preds is not None:
        # Assuming only one face per image, take the first set of landmarks
        landmarks_tensor = torch.tensor(preds[0], dtype=torch.float32)

        # Create a filename for the landmarks tensor based on the image filename
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        landmarks_filename = os.path.join(landmarks_directory, f'{base_filename}_68lms.pt')

        # Save the landmarks tensor
        torch.save(landmarks_tensor, landmarks_filename)
        print(f'Saved landmarks for {os.path.basename(img_path)} to {landmarks_filename}')

# Optional: Display images with landmarks (same as before)
fig, axes = plt.subplots(1, len(image_files), figsize=(5 * len(image_files), 5))
if len(image_files) == 1:
    axes = [axes] # Make it iterable if there's only one image

for i, img_path in enumerate(image_files):
    input_image = io.imread(img_path)
    # Load the saved landmarks for display
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    landmarks_filename = os.path.join(landmarks_directory, f'{base_filename}_68lms.pt')

    axes[i].imshow(input_image)
    if os.path.exists(landmarks_filename):
        landmarks_tensor = torch.load(landmarks_filename)
        landmarks_np = landmarks_tensor.numpy() # Convert back to numpy for plotting
        axes[i].plot(landmarks_np[:, 0], landmarks_np[:, 1], 'o', markersize=2)

    axes[i].set_title(f'Image {os.path.basename(img_path)}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()