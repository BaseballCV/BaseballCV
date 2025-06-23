import os
import subprocess
import sys
import cv2
import random

def extract_frames_from_video(video_path, game_id, output_frames_folder, frames_to_extract) -> list[str]:
    '''
    Extracts frames from a single video file and saves the images into the specified folder.
    '''
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video {os.path.basename(video_path)}...")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_video = min(frames_to_extract, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), frames_video))
    
    extracted_frames = []
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            output_file = os.path.join(output_frames_folder, f"{game_id}_{os.path.splitext(os.path.basename(video_path))[0]}_{i:04d}.jpg")
            cv2.imwrite(output_file, frame)
            extracted_frames.append(output_file)
        else:
            print(f"Could not read frame {frame_index} from video {video_path}")
    
    cap.release()

    return extracted_frames

def check_import(install_path: str, package_name: str) -> bool:
    """
    Checks if a package is installed and attempts to install it if not found.

    Args:
        install_path (str): The path to the package to check.
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_path])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {str(e)}")
            raise
    


