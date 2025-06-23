import os
from typing import Dict, List, Optional, Union
from baseballcv.utilities.environment import check_import 
from baseballcv.model.utils import ModelFunctionUtils
from pkg_resources import resource_filename
from baseballcv.utilities import BaseballCVLogger

class YOLOv9:
    def __init__(self, device: str | int = "cuda", model_path: str = '', cfg_path: str = 'models/detect/yolov9-c.yaml', name: str = 'yolov9-c', custom_weights: bool = False) -> None: 
        """
        Initialize YOLOv9 model.

        Args:
            device (str, optional): Device to use. Defaults to "cpu".
            model_path (str, optional): Path to initial weight. Defaults to ''.
            cfg_path (str, optional): Path to model config. Defaults to 'models/detect/yolov9-c.yaml'.
            name (str, optional): Name of the model. Defaults to 'yolov9-c'.
        """
        from yolov9 import detect_dual as detect, train_dual as train, val_dual as val #lazy load to prevent wandb message on module import
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)
        self.logger.info("Initializing YOLOv9 model...")

        check_import("git+https://github.com/dylandru/yolov9.git", "yolov9")
        self.device = device
        self.name = name
        self.model_path = model_path
        self.model_weights = ModelFunctionUtils.setup_yolo_weights(model_file=f"{name}.pt", output_dir=model_path, custom_weights=custom_weights)
        if not os.path.exists(cfg_path):
            cfg_path = resource_filename('yolov9', cfg_path)
        self.cfg_path = cfg_path
        self.detect = detect
        self.train = train
        self.val = val

        self.logger.info(f"Model initialized: {self.name}")
    
    def finetune(self, data_path: str, epochs: int = 100,
                 imgsz: int = 640, batch_size: int = 16, rect: bool = False, resume: bool = False, nosave: bool = False,
                 noval: bool = False, noautoanchor: bool = False, noplots: bool = False, evolve: Optional[bool] = None,
                 bucket: str = '', cache: Optional[str] = None, image_weights: bool = False, multi_scale: bool = False,
                 single_cls: bool = False, optimizer: str = 'SGD', sync_bn: bool = False, num_workers: int = (os.cpu_count() -1),
                 project: str = 'runs/train', exist_ok: bool = False, quad: bool = False,
                 cos_lr: bool = False, flat_cos_lr: bool = False, fixed_lr: bool = False, label_smoothing: float = 0.0,
                 patience: int = 100, freeze: List[int] = [0], save_period: int = -1, seed: int = 0, local_rank: int = -1,
                 min_items: int = 0, close_mosaic: int = 0, entity: Optional[str] = None, upload_dataset: bool = False,
                 bbox_interval: int = -1) -> Dict:
        
        """
        Finetune a YOLOv9 model.

        Args:
            data_path (str): Path to data config file.
            epochs (int, optional): Number of epochs. Defaults to 100.
            imgsz (int, optional): Image size. Defaults to 640.
            batch_size (int, optional): Batch size. Defaults to 16.
            rect (bool, optional): Rectangular training. Defaults to False.
            resume (bool, optional): Resume training. Defaults to False.
            nosave (bool, optional): Only save final checkpoint. Defaults to False.
            noval (bool, optional): Only validate final epoch. Defaults to False.
            noautoanchor (bool, optional): Disable autoanchor. Defaults to False.
            noplots (bool, optional): Don't save plots. Defaults to False.
            evolve (bool, optional): Evolve hyperparameters. Defaults to None.
            bucket (str, optional): GSUtil bucket. Defaults to ''.
            cache (str, optional): Image caching. Defaults to None.
            image_weights (bool, optional): Use weighted image selection. Defaults to False.
            device (str, optional): Device to use. Defaults to ''.
            multi_scale (bool, optional): Vary img-size ±50%. Defaults to False.
            single_cls (bool, optional): Train as single-class dataset. Defaults to False.
            optimizer (str, optional): Optimizer (SGD, Adam, AdamW, LION). Defaults to 'SGD'.
            sync_bn (bool, optional): Use SyncBatchNorm. Defaults to False.
            num_workers (int, optional): Max dataloader workers. Defaults to 8.
            project (str, optional): Project name. Defaults to 'runs/train'.
            exist_ok (bool, optional): Existing project/name ok. Defaults to False.
            quad (bool, optional): Quad dataloader. Defaults to False.
            cos_lr (bool, optional): Cosine LR scheduler. Defaults to False.
            flat_cos_lr (bool, optional): Flat cosine scheduler. Defaults to False. 
            fixed_lr (bool, optional): Fixed LR scheduler. Defaults to False.
            label_smoothing (float, optional): Label smoothing epsilon. Defaults to 0.0.
            patience (int, optional): EarlyStopping patience. Defaults to 100.
            freeze (List[int], optional): Freeze layers. Defaults to [0].
            save_period (int, optional): Save checkpoint interval. Defaults to -1.
            seed (int, optional): Random seed. Defaults to 0.
            local_rank (int, optional): DDP parameter. Defaults to -1.
            min_items (int, optional): Min items. Defaults to 0.
            close_mosaic (int, optional): Close mosaic. Defaults to 0.
            entity (str, optional): W&B entity. Defaults to None.
            upload_dataset (bool, optional): Upload dataset. Defaults to False.
            bbox_interval (int, optional): Bbox interval. Defaults to -1.
        """

        results = self.train.run(data=data_path, name=self.name, weights=self.model_weights, cfg=self.cfg_path, epochs=epochs, batch_size=batch_size,
                          imgsz=imgsz, rect=rect, resume=resume, nosave=nosave, noval=noval, noautoanchor=noautoanchor,
                          noplots=noplots, evolve=evolve, bucket=bucket, cache=cache, image_weights=image_weights,
                          device=self.device, multi_scale=multi_scale, single_cls=single_cls, optimizer=optimizer,
                          sync_bn=sync_bn, workers=num_workers, project=project, exist_ok=exist_ok, quad=quad, hyp=resource_filename("yolov9", "data/hyps/hyp.scratch-high.yaml"),
                          cos_lr=cos_lr, flat_cos_lr=flat_cos_lr, fixed_lr=fixed_lr, label_smoothing=label_smoothing,
                          patience=patience, freeze=freeze, save_period=save_period, seed=seed, local_rank=local_rank,
                          min_items=min_items, close_mosaic=close_mosaic, entity=entity, upload_dataset=upload_dataset,
                          bbox_interval=bbox_interval)
        
        return results

    def evaluate(self, data_path: str, batch_size: int = 32, imgsz: int = 640, conf_thres: float = 0.001,
                iou_thres: float = 0.7, max_det: int = 300, workers: int = 8, single_cls: bool = False,
                augment: bool = False, verbose: bool = False, save_txt: bool = False, save_hybrid: bool = False,
                save_conf: bool = False, save_json: bool = False, project: str = 'runs/val', exist_ok: bool = False,
                half: bool = True, dnn: bool = False) -> tuple:
        
        """
        Evaluate YOLOv9 model performance on given dataset.

        Args:
            data_path (str): Path to data config file
            batch_size (int, optional): Batch size. Defaults to 32.
            imgsz (int, optional): Image size. Defaults to 640.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.001.
            iou_thres (float, optional): IoU threshold. Defaults to 0.7.
            max_det (int, optional): Maximum detections. Defaults to 300.
            workers (int, optional): Number of workers. Defaults to 8.
            single_cls (bool, optional): Single class. Defaults to False.
            augment (bool, optional): Augment data. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to False.
            save_txt (bool, optional): Save text. Defaults to False.
            save_hybrid (bool, optional): Save hybrid. Defaults to False.
            save_conf (bool, optional): Save confidence. Defaults to False.
            save_json (bool, optional): Save JSON. Defaults to False.
            project (str, optional): Project. Defaults to 'runs/val'.
            exist_ok (bool, optional): Existing project/name ok. Defaults to False.
            half (bool, optional): Half precision. Defaults to True.
            dnn (bool, optional): DNN. Defaults to False.
        """
        results = self.val.run(weights=self.model_weights, data=data_path, name=self.name, batch_size=batch_size, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres,
                         max_det=max_det, workers=workers, single_cls=single_cls, augment=augment, verbose=verbose,
                         save_txt=save_txt, save_hybrid=save_hybrid, save_conf=save_conf, save_json=save_json,
                         project=project, exist_ok=exist_ok, half=half, dnn=dnn, device=self.device)
        return results

    def inference(self, source: Union[str, List[str]], imgsz: tuple = (640, 640), conf_thres: float = 0.25,
                 iou_thres: float = 0.45, max_det: int = 1000, view_img: bool = False, save_txt: bool = False,
                 save_conf: bool = False, save_crop: bool = False, nosave: bool = False,
                 classes: Optional[List[int]] = None, agnostic_nms: bool = False, augment: bool = False,
                 visualize: bool = False, update: bool = False, project: str = 'runs/detect',
                 exist_ok: bool = False, line_thickness: int = 3, hide_labels: bool = False, hide_conf: bool = False,
                 half: bool = False, dnn: bool = False, vid_stride: int = 1) -> List[Dict]:
        """
        Run inference with YOLOv9 model.

        Args:
            source (Union[str, List[str]]): Source path or list of paths
            imgsz (tuple, optional): Image size. Defaults to (640, 640).
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold. Defaults to 0.45.
            max_det (int, optional): Maximum detections. Defaults to 1000.
            view_img (bool, optional): View image. Defaults to False.
            save_txt (bool, optional): Save text. Defaults to False.
            save_conf (bool, optional): Save confidence. Defaults to False.
            save_crop (bool, optional): Save crop. Defaults to False.
            nosave (bool, optional): No save. Defaults to False.
            classes (Optional[List[int]], optional): Classes. Defaults to None.
            agnostic_nms (bool, optional): Agnostic NMS. Defaults to False.
            augment (bool, optional): Augment. Defaults to False.
            visualize (bool, optional): Visualize. Defaults to False.
            update (bool, optional): Update. Defaults to False.
            project (str, optional): Project. Defaults to 'runs/detect'.
            exist_ok (bool, optional): Existing project/name ok. Defaults to False.
            line_thickness (int, optional): Line thickness. Defaults to 3.
            hide_labels (bool, optional): Hide labels. Defaults to False.
            hide_conf (bool, optional): Hide confidence. Defaults to False.
            half (bool, optional): Half precision. Defaults to False.
            dnn (bool, optional): DNN. Defaults to False.
            vid_stride (int, optional): Video stride. Defaults to 1.

        Returns:
            List[Dict]: List of dictionaries containing detection results
        """
        results = self.detect.run(weights=self.model_weights, name=self.name, source=source, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
                           device=self.device, view_img=view_img, save_txt=save_txt, save_conf=save_conf,
                           save_crop=save_crop, nosave=nosave, classes=classes, agnostic_nms=agnostic_nms,
                           augment=augment, visualize=visualize, update=update, project=project,
                           exist_ok=exist_ok, line_thickness=line_thickness, hide_labels=hide_labels,
                           hide_conf=hide_conf, half=half, dnn=dnn, vid_stride=vid_stride)
        return results