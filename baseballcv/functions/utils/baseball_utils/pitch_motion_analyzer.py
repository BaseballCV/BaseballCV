import cv2
import torch
import numpy as np
import supervision as sv
import tempfile
import shutil
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO
from baseballcv.utilities import BaseballCVLogger
from baseballcv.functions.load_tools import LoadTools

class PitchMotionAnalyzer:
    """
    Analyzes a pitcher's motion using video segmentation and detection to find key moments.
    Uses SAM-2 for pitcher segmentation and YOLO models for object detection.
    """
    def __init__(self,
                 model_config: str,
                 model_checkpoint: str,
                 ball_model_path: str = 'ball_trackingv4',
                 pitcher_detector_model: str = 'phc_detector',
                 device: str = 'cpu',
                 verbose: bool = False):
        self.verbose = verbose
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)
        if self.verbose:
            self.logger.set_level('INFO')
        else:
            self.logger.set_level('WARNING')

        self.device = device
        self.load_tools = LoadTools()

        try:
            # Load SAM-2 model using the config path directly
            self.predictor = build_sam2_video_predictor(model_config, model_checkpoint)

            # Move SAM-2 to device
            try:
                if hasattr(self.predictor, 'model'):
                    self.predictor.model.to(device)
                elif hasattr(self.predictor, 'sam'):
                    self.predictor.sam.to(device)
                else:
                    self.predictor = self.predictor.to(device)
            except (AttributeError, Exception) as e:
                self.logger.warning(f"Could not explicitly move SAM2 predictor to {device}: {e}")

            # Load models using LoadTools (proper baseballcv pattern)
            ball_model_path = self.load_tools.load_model(ball_model_path)
            self.ball_model = YOLO(ball_model_path)
            self.ball_model.to(device)
            
            pitcher_model_path = self.load_tools.load_model(pitcher_detector_model)
            self.pitcher_model = YOLO(pitcher_model_path)
            self.pitcher_model.to(device)

            self.logger.info(f"SAM-2, Ball Detection, and Pitcher Detection models loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    @staticmethod
    def _calculate_iou(mask1, mask2):
        """Calculate Intersection over Union of two masks."""
        if mask1 is None or mask2 is None: 
            return 0.0
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        if np.sum(union) == 0: 
            return 1.0
        return np.sum(intersection) / np.sum(union)

    def detect_pitcher_box_multiframe(self, video_path: str, max_frames: int = 30, conf_threshold: float = 0.5) -> tuple:
        """
        Detect pitcher in video by checking multiple frames until found.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to check
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Tuple of (pitcher_box, frame_number) where pitcher was found, or (None, -1) if not found
        """
        self.logger.info(f"Searching for pitcher in first {max_frames} frames...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None, -1
        
        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.pitcher_model.predict(frame, conf=conf_threshold, verbose=False)
            
            pitcher_detections = []
            for result in results:
                for box in result.boxes:
                    if result.names[int(box.cls)] == 'pitcher':
                        pitcher_detections.append({
                            'box': box.xyxy[0].tolist(),
                            'confidence': box.conf[0].item()
                        })
            
            if pitcher_detections:
                # Found pitcher! Return the most confident detection
                pitcher_detections.sort(key=lambda x: x['confidence'], reverse=True)
                best_detection = pitcher_detections[0]
                cap.release()
                self.logger.info(f"Pitcher found in frame {frame_idx} with confidence {best_detection['confidence']:.3f}")
                return best_detection['box'], frame_idx
            
            frame_idx += 1
        
        cap.release()
        self.logger.error(f"No pitcher detected in first {max_frames} frames")
        return None, -1

    def detect_pitcher_box(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list:
        """
        Detect pitcher in a single frame using the PHC detector model.
        
        Args:
            frame: Input video frame
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Bounding box [x1, y1, x2, y2] of the pitcher, or None if not found
        """
        results = self.pitcher_model.predict(frame, conf=conf_threshold, verbose=False)
        
        pitcher_detections = []
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == 'pitcher':
                    pitcher_detections.append({
                        'box': box.xyxy[0].tolist(),
                        'confidence': box.conf[0].item()
                    })
        
        if not pitcher_detections:
            return None
            
        # Return the most confident detection
        pitcher_detections.sort(key=lambda x: x['confidence'], reverse=True)
        return pitcher_detections[0]['box']

    def find_motion_start(self, video_path: str, initial_box: list = None, iou_threshold: float = 0.95, 
                         frame_buffer: int = 5, debug_viz_path: str = None, 
                         create_overlay_video: bool = True) -> int:
        """
        Find when the pitcher starts moving using SAM-2 segmentation.
        
        Args:
            video_path: Path to the video file
            initial_box: Initial bounding box [x1, y1, x2, y2] for pitcher, if None will auto-detect
            iou_threshold: IoU threshold below which motion is detected
            frame_buffer: Number of frames to skip before detecting motion
            debug_viz_path: Path to save debug visualizations
            create_overlay_video: Whether to create a video with segmentation overlay
            
        Returns:
            Frame index where motion starts
        """
        self.logger.info("Analyzing video to find motion start...")
        temp_dir = tempfile.mkdtemp(prefix="pitch_motion_")
        
        try:
            # Extract frames
            frames_generator = sv.get_video_frames_generator(source_path=video_path)
            image_sink = sv.ImageSink(target_dir_path=temp_dir, image_name_pattern="{:05d}.jpeg", overwrite=True)
            files_written = 0
            frames_list = []  # Store frames for overlay video
            
            with image_sink as sink:
                for frame in frames_generator:
                    if frame is None:
                        self.logger.warning("A null frame was skipped during video processing.")
                        continue
                    sink.save_image(frame)
                    if create_overlay_video:
                        frames_list.append(frame.copy())
                    files_written += 1
            
            if files_written == 0:
                self.logger.error(f"Failed to extract any frames from {video_path}. The video may be corrupt.")
                raise ValueError(f"No frames could be extracted from {video_path}")

            self.logger.info(f"{files_written} frames extracted to temporary directory: {temp_dir}")
            
            # Auto-detect pitcher if no box provided (check multiple frames)
            if initial_box is None:
                initial_box, pitcher_frame = self.detect_pitcher_box_multiframe(video_path)
                
                if initial_box is None:
                    self.logger.error("Could not detect pitcher in any of the first frames")
                    raise ValueError("Pitcher detection failed")
                
                self.logger.info(f"Auto-detected pitcher box: {initial_box}")
            
            # Initialize SAM-2
            inference_state = self.predictor.init_state(video_path=temp_dir)
            
            # Use center of bounding box as the point prompt
            center_x = (initial_box[0] + initial_box[2]) / 2
            center_y = (initial_box[1] + initial_box[3]) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            _, _, mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1, 
                points=center_point,
                labels=point_labels,
            )
            prev_mask = (mask_logits > 0.0).cpu().numpy().squeeze()

            # Save initial debug visualization
            if debug_viz_path:
                frame_zero_path = os.path.join(temp_dir, "00000.jpeg")
                if os.path.exists(frame_zero_path):
                    frame_zero = cv2.imread(frame_zero_path)
                    annotated_frame = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(
                        scene=frame_zero.copy(), 
                        detections=sv.Detections(
                            xyxy=sv.mask_to_xyxy(masks=np.array([prev_mask])), 
                            mask=np.array([prev_mask])
                        )
                    )
                    # Draw the initial box and center point
                    cv2.rectangle(annotated_frame, (int(initial_box[0]), int(initial_box[1])), 
                                (int(initial_box[2]), int(initial_box[3])), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(debug_viz_path, "initial_mask.jpg"), annotated_frame)

            # Track motion through frames
            self.logger.info("Tracking pitcher's motion frame by frame using SAM-2 segmentation...")
            iou_scores = []
            motion_detected_frame = -1
            overlay_frames = []
            
            # Store the current mask for ball release detection later
            self.current_pitcher_mask = None
            
            for frame_idx, _, mask_logits in self.predictor.propagate_in_video(inference_state):
                current_mask = (mask_logits > 0.0).cpu().numpy().squeeze()
                iou = self._calculate_iou(prev_mask, current_mask)
                iou_scores.append((frame_idx, iou))
                
                if self.verbose:
                    self.logger.info(f"Frame {frame_idx}: IoU with previous frame = {iou:.4f}")

                # Create overlay frame if requested
                if create_overlay_video and frame_idx < len(frames_list):
                    overlay_frame = frames_list[frame_idx].copy()
                    mask_overlay = sv.MaskAnnotator(
                        color_lookup=sv.ColorLookup.INDEX,
                        opacity=0.3
                    ).annotate(
                        scene=overlay_frame, 
                        detections=sv.Detections(
                            xyxy=sv.mask_to_xyxy(masks=np.array([current_mask])), 
                            mask=np.array([current_mask])
                        )
                    )
                    overlay_frames.append(mask_overlay)

                # Detect significant motion
                if iou < iou_threshold and motion_detected_frame == -1 and frame_idx > frame_buffer:
                    self.logger.info(f"Significant motion detected at frame {frame_idx} (IoU: {iou:.4f})")
                    motion_detected_frame = frame_idx
                    
                    if debug_viz_path:
                        frame_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpeg")
                        if os.path.exists(frame_path):
                            frame = cv2.imread(frame_path)
                            annotated_frame = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(
                                scene=frame.copy(), 
                                detections=sv.Detections(
                                    xyxy=sv.mask_to_xyxy(masks=np.array([current_mask])), 
                                    mask=np.array([current_mask])
                                )
                            )
                            cv2.imwrite(os.path.join(debug_viz_path, "motion_detection_frame.jpg"), annotated_frame)
                
                prev_mask = current_mask
                
                # Store the last mask for ball release detection
                self.current_pitcher_mask = current_mask

            # Save overlay video if requested
            if create_overlay_video and overlay_frames and debug_viz_path:
                self.logger.info("Creating segmentation overlay video...")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                overlay_video_path = os.path.join(debug_viz_path, "segmentation_overlay.mp4")
                height, width = overlay_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
                
                for frame in overlay_frames:
                    out.write(frame)
                out.release()
                self.logger.info(f"Segmentation overlay video saved to: {overlay_video_path}")

            # Save IoU plot
            if debug_viz_path and iou_scores:
                frames, ious = zip(*iou_scores)
                plt.figure(figsize=(12, 6))
                plt.plot(frames, ious, marker='o', linestyle='-', markersize=3)
                plt.axhline(y=iou_threshold, color='r', linestyle='--', 
                           label=f'Motion Threshold ({iou_threshold})')
                if motion_detected_frame != -1:
                    plt.axvline(x=motion_detected_frame, color='g', linestyle='--', 
                               label=f'Motion Start (Frame {motion_detected_frame})')
                plt.title("Pitcher Segmentation IoU Over Time")
                plt.xlabel("Frame Index")
                plt.ylabel("Intersection over Union (IoU)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(debug_viz_path, "iou_plot.png"), dpi=150)
                plt.close()

            if motion_detected_frame != -1:
                return motion_detected_frame

            self.logger.warning("No significant motion start detected. Returning frame 5 as default.")
            return 5
            
        finally:
            shutil.rmtree(temp_dir)
            self.logger.info(f"Cleaned up temporary directory: {temp_dir}")

    def find_ball_release(self, video_path: str, start_frame: int, pitcher_box: list = None, 
                         min_consecutive_detections: int = 3, min_frames_outside_pitcher: int = 2,
                         debug_viz_path: str = None) -> int:
        """
        Find when the ball is released using robust detection criteria:
        1. Ball must be consistently detected for several frames
        2. Ball must be outside the pitcher's segmentation mask
        3. This provides much more reliable detection than velocity alone
        
        Args:
            video_path: Path to the video file
            start_frame: Frame to start looking for ball release
            pitcher_box: Bounding box of pitcher (for filtering detections)
            min_consecutive_detections: Minimum consecutive frames ball must be detected
            min_frames_outside_pitcher: Minimum frames ball must be outside pitcher mask
            debug_viz_path: Path to save debug visualizations
            
        Returns:
            Frame index where ball release occurs
        """
        self.logger.info(f"Analyzing video to find ball release starting from frame {start_frame}...")
        self.logger.info(f"Using robust detection: {min_consecutive_detections} consecutive detections, {min_frames_outside_pitcher} frames outside pitcher")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return start_frame + 50

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        
        # Track ball detection state
        consecutive_detections = 0
        frames_outside_pitcher = 0
        ball_positions = []
        detection_history = []
        
        # Get video properties for debug video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        debug_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Run ball detection
            results = self.ball_model.predict(frame, verbose=False)
            result = results[0]
            
            ball_detections = []
            for box in result.boxes:
                label = result.names[int(box.cls)]
                if label == 'baseball':
                    ball_detections.append({
                        'box': box.xyxy[0].tolist(), 
                        'confidence': box.conf[0].item(),
                        'center': [(box.xyxy[0][0] + box.xyxy[0][2]) / 2, 
                                  (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                    })

            current_ball_pos = None
            is_outside_pitcher = False
            
            if ball_detections:
                # Use the most confident detection
                ball_detections.sort(key=lambda x: x['confidence'], reverse=True)
                best_ball = ball_detections[0]
                current_ball_pos = best_ball['center']
                
                # Check if ball is outside pitcher mask (if we have one)
                if hasattr(self, 'current_pitcher_mask') and self.current_pitcher_mask is not None:
                    ball_x, ball_y = int(current_ball_pos[0]), int(current_ball_pos[1])
                    
                    # Check if ball center is outside the pitcher mask
                    if (0 <= ball_x < self.current_pitcher_mask.shape[1] and 
                        0 <= ball_y < self.current_pitcher_mask.shape[0]):
                        is_outside_pitcher = not self.current_pitcher_mask[ball_y, ball_x]
                    else:
                        is_outside_pitcher = True  # Ball is outside frame bounds
                else:
                    # Fallback: check if ball is outside pitcher bounding box
                    if pitcher_box:
                        is_outside_pitcher = not (pitcher_box[0] <= current_ball_pos[0] <= pitcher_box[2] and
                                                pitcher_box[1] <= current_ball_pos[1] <= pitcher_box[3])
                    else:
                        is_outside_pitcher = True  # Assume outside if no reference
                
                # Update tracking state
                consecutive_detections += 1
                if is_outside_pitcher:
                    frames_outside_pitcher += 1
                else:
                    frames_outside_pitcher = 0  # Reset if ball goes back inside
                
                ball_positions.append((frame_idx, current_ball_pos, is_outside_pitcher))
                
                # Check for ball release conditions
                if (consecutive_detections >= min_consecutive_detections and 
                    frames_outside_pitcher >= min_frames_outside_pitcher):
                    
                    self.logger.info(f"Ball release detected at frame {frame_idx}")
                    self.logger.info(f"  - Consecutive detections: {consecutive_detections}")
                    self.logger.info(f"  - Frames outside pitcher: {frames_outside_pitcher}")
                    
                    # Save debug frame if path provided
                    if debug_viz_path:
                        debug_frame = frame.copy()
                        box = best_ball['box']
                        cv2.rectangle(debug_frame, (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), (0, 0, 255), 3)
                        cv2.putText(debug_frame, f"RELEASE! Frame {frame_idx}", 
                                  (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 0, 255), 2)
                        cv2.putText(debug_frame, f"Consecutive: {consecutive_detections}", 
                                  (int(box[0]), int(box[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 1)
                        cv2.putText(debug_frame, f"Outside: {frames_outside_pitcher}", 
                                  (int(box[0]), int(box[1]-50)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 1)
                        cv2.imwrite(os.path.join(debug_viz_path, "ball_release_frame.jpg"), debug_frame)
                    
                    cap.release()
                    return frame_idx
                    
            else:
                # No ball detected, reset consecutive counter
                consecutive_detections = 0
                frames_outside_pitcher = 0

            # Store detection history for debugging
            detection_history.append({
                'frame': frame_idx,
                'detected': current_ball_pos is not None,
                'outside_pitcher': is_outside_pitcher,
                'consecutive': consecutive_detections,
                'outside_count': frames_outside_pitcher
            })

            # Store debug frame
            if debug_viz_path:
                debug_frame = frame.copy()
                if current_ball_pos is not None:
                    color = (0, 255, 0) if is_outside_pitcher else (0, 255, 255)  # Green if outside, yellow if inside
                    cv2.circle(debug_frame, (int(current_ball_pos[0]), int(current_ball_pos[1])), 
                             5, color, -1)
                    cv2.putText(debug_frame, f"Det:{consecutive_detections} Out:{frames_outside_pitcher}", 
                              (int(current_ball_pos[0]+10), int(current_ball_pos[1])), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                debug_frames.append(debug_frame)

            frame_idx += 1
            
            # Stop searching after reasonable number of frames
            if frame_idx > start_frame + 150:
                self.logger.warning("Ball release not detected within 150 frames of motion start.")
                break
        
        cap.release()
        
        # Save debug visualizations
        if debug_viz_path and detection_history:
            # Plot detection history
            frames = [d['frame'] for d in detection_history]
            detected = [1 if d['detected'] else 0 for d in detection_history]
            outside = [1 if d['outside_pitcher'] else 0 for d in detection_history]
            consecutive = [d['consecutive'] for d in detection_history]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
            
            # Detection status
            ax1.plot(frames, detected, 'bo-', markersize=3, label='Ball Detected')
            ax1.set_ylabel('Detected')
            ax1.set_title('Ball Detection Status Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Outside pitcher status
            ax2.plot(frames, outside, 'ro-', markersize=3, label='Outside Pitcher')
            ax2.axhline(y=min_frames_outside_pitcher, color='r', linestyle='--', 
                       label=f'Required Outside Frames ({min_frames_outside_pitcher})')
            ax2.set_ylabel('Outside Pitcher')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Consecutive detections
            ax3.plot(frames, consecutive, 'go-', markersize=3, label='Consecutive Detections')
            ax3.axhline(y=min_consecutive_detections, color='g', linestyle='--', 
                       label=f'Required Consecutive ({min_consecutive_detections})')
            ax3.set_ylabel('Consecutive Count')
            ax3.set_xlabel('Frame Index')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(debug_viz_path, "ball_detection_analysis.png"), dpi=150)
            plt.close()
        
        # Return last detected frame or estimate
        self.logger.warning(f"Ball release detection failed, returning frame {frame_idx}")
        return frame_idx

    def trim_pitching_motion(self, video_path: str, output_path: str, pitcher_box: list = None, 
                           end_frame_offset: int = 15, debug_viz_path: str = None,
                           create_overlay_video: bool = True):
        """
        Trim video to contain only the pitching motion.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            pitcher_box: Initial pitcher bounding box (will auto-detect if None)
            end_frame_offset: Frames to include after ball release
            debug_viz_path: Path to save debug files
            create_overlay_video: Whether to create segmentation overlay video
        """
        # Find motion start using segmentation
        start_frame = self.find_motion_start(
            video_path=video_path, 
            initial_box=pitcher_box, 
            debug_viz_path=debug_viz_path,
            create_overlay_video=create_overlay_video
        )
        
        # Find ball release using robust detection
        release_frame = self.find_ball_release(
            video_path=video_path, 
            start_frame=start_frame, 
            pitcher_box=pitcher_box,
            debug_viz_path=debug_viz_path
        )
        
        end_frame = release_frame + end_frame_offset
        
        # Trim the video
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
        frames_written = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            if start_frame <= frame_idx <= end_frame:
                out.write(frame)
                frames_written += 1
                
            frame_idx += 1
            if frame_idx > end_frame: 
                break
        
        cap.release()
        out.release()
        
        self.logger.info(f"Video trimmed from frame {start_frame} to {end_frame} ({frames_written} frames). Saved to {output_path}")
        
        # Save summary info
        if debug_viz_path:
            summary = {
                'start_frame': start_frame,
                'release_frame': release_frame,
                'end_frame': end_frame,
                'total_frames': frames_written,
                'duration_seconds': frames_written / fps
            }
            
            with open(os.path.join(debug_viz_path, "trim_summary.txt"), 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")