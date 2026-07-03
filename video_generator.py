import cv2
import os
import glob
import re
from datetime import datetime
# Specify the directory containing the images
image_folder = '/home/erdi/Storage/publications/erdi_dpcc_test/dpcc_original_t1/logs/0/diffmpc_0/diffusion/'

images = sorted(glob.glob(f'{image_folder}/*.png'))


# def extract_timestamp(filename):
#     base = os.path.basename(filename)
#     timestamp_str = base.replace('diffusion_', '').replace('.png', '')
#     return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S_%f')

# images = sorted(image_folder, key=extract_timestamp)

# Sort filenames numerically
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else 0

# Get sorted file paths
# images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=numerical_sort)

# Check if images are found
if not images:
    raise ValueError("No images found in the directory.")

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define video codec and create VideoWriter object
video_name = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 19, (width, height))

# Loop through each image and write to the video
for image_path in images:
    frame = cv2.imread(image_path)
    video.write(frame)

video.release()


import subprocess
import os


input_file = "output.mp4"
temp_output = "output_temp.mp4"

# FFmpeg command with -y to auto-overwrite temp file
ffmpeg_command = [
    "ffmpeg",
    "-y",  # Overwrite without asking
    "-i", input_file,
    "-vcodec", "libx264",
    "-acodec", "aac",
    temp_output
]

try:
    subprocess.run(ffmpeg_command, check=True)
    
    # Replace original file with converted one
    os.replace(temp_output, input_file)
    
    print(f"Video re-encoded and saved as {input_file}")
except subprocess.CalledProcessError as e:
    print("Error during conversion:", e)
    # Optionally remove temp file if exists
    if os.path.exists(temp_output):
        os.remove(temp_output)


print(f"Video saved as {video_name}")
