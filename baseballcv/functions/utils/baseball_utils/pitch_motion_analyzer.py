import cv2
import torch
import numpy as np
import supervision as sv
import tempfile
import shutil
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
from baseballcv.model.od.yolo import YOLOModel
from baseballcv.utilities import BaseballCVLogger

class PitchMotionAnalyzer:
    """
    Analyzes a pitcher's motion using video segmentation and detection to find key moments.
    """
    def __init__(self, 
                 model_config: str, 
                 model_checkpoint: str, 
                 ball_model_path: str = 'ball_trackingv4',
                 device: str = 'cpu', 
                 verbose: bool = False):
        """
        Initializes the PitchMotionAnalyzer.

        Args:
            model_config (str): Path to the SAM-2 model config file.
            model_checkpoint (str): Path to the SAM-2 model checkpoint file.
            ball_model_path (str): Model alias/path for the ball detection model.
            device (str): Device to run the model on ('cpu', 'cuda').
            verbose (bool): If True, prints detailed logs.
        """
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__, verbose)
        self.device = device
        self.verbose = verbose
        
        try:
            self.predictor = build_sam2_video_predictor(model_config, model_checkpoint)
            self.predictor.model.to(device)
            self.ball_model = YOLOModel(model_path=ball_model_path, task='detect')
            self.ball_model.model.to(device)
            self.logger.info(f"SAM-2 and Ball Detection models loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    @staticmethod
    def _calculate_iou(mask1, mask2):
        """Calculates the Intersection over Union (IoU) of two boolean masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        if np.sum(union) == 0:
            return 1.0
        return np.sum(intersection) / np.sum(union)

    def find_motion_start(self, video_path: str, initial_box: list, iou_threshold: float = 0.97, frame_buffer: int = 3) -> int:
        """
        Finds the frame index where the pitcher's motion starts by detecting changes in the segmentation mask.
        """
        self.logger.info("Analyzing video to find motion start...")
        temp_dir = tempfile.mkdtemp(prefix="pitch_motion_")
        
        try:
            frames_generator = sv.get_video_frames_generator(video_path=video_path)
            image_sink = sv.ImageSink(target_dir_path=temp_dir, overwrite=True)
            with image_sink as sink:
                for frame in frames_generator:
                    sink.save_image(frame)
            
            self.logger.info(f"Frames extracted to temporary directory: {temp_dir}")

            inference_state = self.predictor.init_state(video_path=temp_dir)
            
            box_prompt = torch.tensor([initial_box], device=self.device)
            _, _, mask_logits = self.predictor.add_new_prompts(
                inference_state=inference_state, frame_idx=0, prompts={"bboxes": box_prompt}
            )
            
            prev_mask = (mask_logits > 0.0).cpu().numpy().squeeze()

            self.logger.info("Tracking pitcher's motion frame by frame...")
            for frame_idx, _, mask_logits in self.predictor.propagate_in_video(inference_state):
                if frame_idx <= frame_buffer:
                    prev_mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                    continue

                current_mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                
                iou = self._calculate_iou(prev_mask, current_mask)
                if self.verbose:
                    self.logger.info(f"Frame {frame_idx}: IoU with previous frame = {iou:.4f}")

                if iou < iou_threshold:
                    self.logger.info(f"Significant motion detected at frame {frame_idx}.")
                    return frame_idx
                
                prev_mask = current_mask
            
            self.logger.warning("No significant motion start detected. Returning frame 0.")
            return 0
        finally:
            shutil.rmtree(temp_dir)
            self.logger.info(f"Cleaned up temporary directory: {temp_dir}")

    def find_ball_release(self, video_path: str, start_frame: int, pitcher_mask, velocity_threshold: int = 20) -> int:
        """
        Finds the frame index where the ball is released by tracking its velocity.

        Args:
            video_path (str): Path to the video file.
            start_frame (int): The frame index where the pitching motion starts.
            pitcher_mask (np.ndarray): The segmentation mask of the pitcher to help isolate the ball.
            velocity_threshold (int): The change in pixels per frame to trigger release detection.

        Returns:
            int: The frame index where the ball is released.
        """
        self.logger.info("Analyzing video to find ball release...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return start_frame + 50 # Return a default offset

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        last_ball_pos = None
        frame_idx = start_frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.ball_model.predict(frame)
            ball_detections = [d for d in detections if d['label'] == 'baseball']
            
            if ball_detections:
                ball_detections.sort(key=lambda x: x['confidence'], reverse=True)
                box = ball_detections[0]['box']
                current_ball_pos = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

                if last_ball_pos is not None:
                    velocity = np.linalg.norm(current_ball_pos - last_ball_pos)
                    if self.verbose:
                        self.logger.info(f"Frame {frame_idx}: Ball velocity: {velocity:.2f} pixels/frame")
                    
                    if velocity > velocity_threshold:
                        self.logger.info(f"Ball release detected at frame {frame_idx} with velocity {velocity:.2f}")
                        cap.release()
                        return frame_idx
                
                last_ball_pos = current_ball_pos
            else:
                last_ball_pos = None # Reset if ball is not detected

            frame_idx += 1
            # Add a timeout to avoid searching the whole video
            if frame_idx > start_frame + 100:
                self.logger.warning("Ball release not detected within 100 frames of motion start.")
                break
        
        cap.release()
        return frame_idx

    def trim_pitching_motion(self, video_path: str, output_path: str, pitcher_box: list):
        """
        Identifies pitcher motion start and ball release to trim the video.
        """
        start_frame = self.find_motion_start(video_path=video_path, initial_box=pitcher_box)
        
        # We pass the initial pitcher box to help narrow down the search for the ball later if needed.
        # This implementation of find_ball_release doesn't use the mask yet, but it's good practice.
        release_frame = self.find_ball_release(video_path=video_path, start_frame=start_frame, pitcher_mask=None)
        
        # Add a small buffer after release
        end_frame = release_frame + 5
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= start_frame and frame_idx <= end_frame:
                out.write(frame)
            
            frame_idx += 1
            if frame_idx > end_frame:
                break
        
        cap.release()
        out.release()
        self.logger.info(f"Video trimmed from frame {start_frame} to {end_frame}. Saved to {output_path}")