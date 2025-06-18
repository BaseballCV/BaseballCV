from supervision import DetectionDataset, Detections
from supervision.detection.utils import polygon_to_mask
from supervision.dataset.utils import save_dataset_images
import os
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
import random
import json
import re
from typing import Tuple, List, Optional, Dict, Union
import math # For floating point rounding issues
from baseballcv.utilities import BaseballCVLogger
import glob

# What's left to do:
# 1. Implement functionality to save jsonl files
# 2. Implement Train, test, val, split functionality
# 3. Fix dumb bug for the masking

def jsonl_to_detections(image_annotations: str, 
                        resolution_wh: Tuple[int, int], with_masks: bool,
                        classes: Dict[str, int]) -> Detections:
        
        if not image_annotations:
            return Detections.empty()
        
        w, h = resolution_wh

        if w <= 0 and h <=0:
            raise ValueError(f'Both dimensions must be positive. Got width {w} and height {h}')

        pattern = re.compile(r"(?<!<loc\d{4}>)<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})> ([\w\s\-]+)")
        matches = pattern.findall(image_annotations)

        matches = np.array(matches) if matches else np.empty((0, 5))

        xyxy, class_name = matches[:, [1, 0, 3, 2]], matches[:, 4]
        xyxy = xyxy.astype(int) / 1024 * np.array([w, h, w, h])
        class_name = np.char.strip(class_name.astype(str))

        filter = np.array([name in classes for name in class_name], dtype=bool)
        xyxy = xyxy[filter]
        class_name = class_name[filter]
        class_id = np.array([classes.get(name) for name in class_name])

        relative_polygons = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for (x1, y1, x2, y2) in xyxy]

        if with_masks:
            polygons = [
                    (polygon * np.array(resolution_wh)).astype(int) for polygon in relative_polygons
                ]
            mask = np.array(
                [
                    polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
                    for polygon in polygons
                ],
                dtype=bool,
            )
            
            if not np.any(mask):
                # This is some weird bug that's caused by the whol array being False. Need to figure out why.
                return Detections.empty()

            return Detections(xyxy=xyxy, class_id=class_id, mask=mask)

        return Detections(xyxy=xyxy, class_id=class_id)

def read_jsonl(path: str) -> List[dict]:
        data = []
        with open(str(path), 'r') as f:
            json_lines = list(f)

        for json_line in json_lines:
            result = json.loads(json_line)
            data.append(result)
        
        return data

def detections_to_jsonl_annotations(
        detections: Detections, image_shape: Tuple[int, int],
        image_name: str, min_image_area_percentage: float, max_image_area_percentage: float,
        approximation_percentage: float
        ):
    pass

def save_jsonl_file(lines: list, file_path: str) -> None:
    pass

class NewDetectionsDataset(DetectionDataset):
    """
    A Monkey Patch Class of Roboflow's `DetectionDataset` with a JSONL implementation in place
    """

    def __init__(self, classes, images, annotations):
        super().__init__(classes, images, annotations)

    @classmethod
    def from_jsonl(cls, images_directory_path: str, annotations_path: str, force_masks: bool) -> DetectionDataset:
        jsonl_data = read_jsonl(path=annotations_path)

        images = []
        annotations = {}

        # assume prefix is the same throughout JSONL
        classes_dict = None
        assigned_class = False

        for jsonl_image in jsonl_data:
            # Extract name, width, height from the name + suffix
            image_name = jsonl_image['image']

            image_path = os.path.join(images_directory_path, image_name)

            (image_width, image_height, _) = cv2.imread(image_path).shape

            pattern = re.compile(r'\b(?!detect\b)(\w+)')

            classes = pattern.findall(jsonl_image['prefix'])

            if assigned_class == False:
                classes_dict = {name: identifier for identifier, name in enumerate(classes)}
                assigned_class = True


            annotation = jsonl_to_detections(
                image_annotations=jsonl_image['suffix'],
                resolution_wh=(image_width, image_height),
                with_masks=force_masks,
                classes=classes_dict
            )

            if annotation.is_empty() and force_masks:
                continue
            
            images.append(image_path)
            annotations[image_path] = annotation

        return DetectionDataset(classes=classes, images=images, annotations=annotations)
    
    def as_jsonl(self: "NewDetectionsDataset",
        images_directory_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        if images_directory_path:
            save_dataset_images(
                dataset=self, 
                images_directory_path=images_directory_path
            )
        
        if annotations_path:
            Path(annotations_path).mkdir(parents=True, exist_ok=True)

            lines = []
            for image_path, image, annotation in self:
                image_name = Path(image_path).name

                line = detections_to_jsonl_annotations(
                    detections=annotation,
                    image_shape = image.shape,
                    image_name=image_name,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage
                )

                lines.append(line)
            
            save_jsonl_file(lines=lines, file_path=annotations_path)
            
    
class _BaseFmt:

    def __init__(self, params):
        self.params = params
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

        dir_list = os.listdir(self.params.root_dir)


        train_dir = self._find_respective_file(dir_list, 'train')
        test_dir = self._find_respective_file(dir_list, 'test')
        val_dir = self._find_respective_file(dir_list, 'val')
        annotations_dir = self._find_respective_file(dir_list, 'annotation')
        dataset_dir = self._find_respective_file(dir_list, 'dataset')

        # TODO: Address this statement for JSONL
        # if train_dir is None:
        #     raise FileNotFoundError('There needs to at least be a `train` folder')

        dir_attrs = {
            "train_dir": train_dir,
            "test_dir": test_dir,
            "val_dir": val_dir,
            "annotations_dir": annotations_dir,
            "dataset_dir": dataset_dir
        }

        for attr_name, dir_name in dir_attrs.items():
            if dir_name is not None:
                full_path = os.path.join(self.params.root_dir, dir_name)
                setattr(self, attr_name, full_path)

        self.new_dir = f"{self.params.root_dir}_conversion"
        os.makedirs(self.new_dir, exist_ok=True)


    @property
    def detections_data(self): raise NotImplementedError

    def to_coco(self, detections_data: tuple):
        train_det, test_det, val_det = detections_data

        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'train'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_train.json'))
        
        if test_det and val_det:
            test_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'test'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_test.json'))
            val_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'val'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_val.json'))
            
    def to_yolo(self, detections_data: tuple):
        train_det, test_det, val_det = detections_data
        
        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_yolo(images_directory_path=os.path.join(self.new_dir, 'train', 'images'),
                          annotations_directory_path=os.path.join(self.new_dir, 'train', 'labels'),
                          data_yaml_path=os.path.join(self.new_dir, 'train_detections'))
        
        if test_det and val_det:
            test_det.as_yolo(images_directory_path=os.path.join(self.new_dir, 'test', 'images'),
                          annotations_directory_path=os.path.join(self.new_dir, 'test', 'labels'),
                          data_yaml_path=os.path.join(self.new_dir, 'test_detections'))
            
            val_det.as_yolo(images_directory_path=os.path.join(self.new_dir, 'val', 'images'),
                          annotations_directory_path=os.path.join(self.new_dir, 'val', 'labels'),
                          data_yaml_path=os.path.join(self.new_dir, 'val_detections'))

    def to_pascal(self, detections_data: tuple): 
        train_det, test_det, val_det = detections_data

        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'train', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'train', 'labels'))
        
        if test_det and val_det:
            test_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'test', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'test', 'labels'))
            val_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'val', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'val', 'labels'))

    def to_jsonl(self, detections_data: tuple):
        train_det, test_det, val_det = detections_data

        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_jsonl(images_directory_path=os.path.join(self.new_dir, 'train'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'annotations', 'annotations_train.jsonl'))
        
        if test_det and val_det:
            test_det.as_jsonl(images_directory_path=os.path.join(self.new_dir, 'test'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'annotations', 'annotations_test.jsonl'))
            val_det.as_jsonl(images_directory_path=os.path.join(self.new_dir, 'val'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'annotations', 'annotations_val.jsonl'))

    def _find_respective_file(self, dir_list: List[str], query: str) -> Optional[str]:
        for item in dir_list:
            if query in item and os.path.isdir(os.path.join(self.params.root_dir, item)):
                return item
        return None

    def _train_test_val_split(self, data: List[str], split: Tuple[int, int, int], 
                              random_state: int=None, shuffle: bool=True,
                              stratify = None
                              ) -> Tuple[List, List, List]:
        """Assumes there is only a train input
        A future implementation could be stratifying based on the classname.

        Args:
            data (List[str]): _description_
            split (Tuple[int, int, int]): _description_
            random_state (int, optional): _description_. Defaults to None.
            shuffle (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            Tuple[List, List, List]: _description_
        """

        if not math.isclose(sum(split), 1.0, rel_tol=1e-9):
            raise ValueError('The split should sum to 1')
        
        if len(split) != 3 or not all(split):
            raise IndexError('There must be 3 entries in the split: train, test, val.')

        if random_state is not None:
            random.seed(random_state)
        
        if shuffle:
            random.shuffle(data)

        if stratify:
            raise ValueError('Not supporting stratify right now')

        train_end = int(round(len(data)* split[0]))
        test_end = train_end + int(round(len(data) * split[1]))
        val_end = test_end + int(round(len(data) * split[2]))

        return data[:train_end], data[train_end:test_end], data[test_end:val_end]
    
    def _validate_all_splits(self) -> bool:
        """
        Validates if all the files are found and if not, only train file is focused on.

        Returns:
            bool: _description_
        """
 
        return all([hasattr(self, 'train_dir'), hasattr(self, 'test_dir'), hasattr(self, 'val_dir')])
    
class YOLOFmt(_BaseFmt):
    @property
    def detections_data(self):

        train_detections, test_detections, val_detections = (None, None, None)

        try:
            yaml_pth = glob.glob(os.path.join(self.params.root_dir, '**', '*.y?ml'), recursive=True)[0]
        except IndexError:
            self.logger.error('Make sure you have a specified yaml file in your directory.')

        train_detections = NewDetectionsDataset.from_yolo(
            images_directory_path=os.path.join(self.train_dir, 'images'),
            annotations_directory_path=os.path.join(self.train_dir, 'labels'),
            data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
            )

        if self._validate_all_splits():
            test_detections = NewDetectionsDataset.from_yolo(
                images_directory_path=os.path.join(self.test_dir, 'images'),
                annotations_directory_path=os.path.join(self.test_dir, 'labels'),
                data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
                )

            val_detections = NewDetectionsDataset.from_yolo(
                images_directory_path=os.path.join(self.val_dir, 'images'),
                annotations_directory_path=os.path.join(self.val_dir, 'labels'),
                data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
                )
        return (train_detections, test_detections, val_detections)
        
    def to_coco(self):
        super().to_coco(detections_data=self.detections_data)

    def to_pascal(self):
        super().to_pascal(detections_data=self.detections_data)

    def to_jsonl(self):
        super().to_jsonl(detections_data=self.detections_data)
        
class COCOFmt(_BaseFmt):
    # Requires self.annotations_dir so if not exists, raise valueerror
    @property
    def detections_data(self):

        train_detections, test_detections, val_detections = (None, None, None)
        # TODO: Need to check for other json file names
        if not hasattr(self, 'annotations_dir'):
            self.logger.error('There needs to be an annotations directory containing the .json files')
            return (train_detections, test_detections, val_detections )

        train_detections = NewDetectionsDataset.from_coco(
            images_directory_path=os.path.join(self.train_dir, 'images'),
            annotations_path=os.path.join(self.annotations_dir, 'instances_train.json'),
            force_masks=self.params.force_masks
            )

        if self._validate_all_splits():
            test_detections = NewDetectionsDataset.from_coco(
                images_directory_path=os.path.join(self.test_dir, 'images'),
                annotations_path=os.path.join(self.annotations_dir, 'instances_test.json'),
                force_masks=self.params.force_masks
                )

            val_detections = NewDetectionsDataset.from_coco(
                images_directory_path=os.path.join(self.val_dir, 'images'),
                annotations_path=os.path.join(self.annotations_dir, 'instances_val.json'),
                force_masks=self.params.force_masks
                )
        return (train_detections, test_detections, val_detections)
        
    def to_yolo(self):
        super().to_yolo(detections_data=self.detections_data)

    def to_pascal(self):
        super().to_pascal(detections_data=self.detections_data)

    def to_jsonl(self):
        super().to_jsonl(detections_data=self.detections_data)

class PascalFmt(_BaseFmt):
    @property
    def detections_data(self):

        train_detections, test_detections, val_detections = (None, None, None)

        train_detections = NewDetectionsDataset.from_pascal_voc(
            images_directory_path=self.train_dir,
            annotations_path=self.train_dir,
            force_masks=self.params.force_masks
            )

        if self._validate_all_splits():
            test_detections = NewDetectionsDataset.from_pascal_voc(
                images_directory_path=self.test_dir,
                annotations_path=self.test_dir,
                force_masks=self.params.force_masks
                )

            val_detections = NewDetectionsDataset.from_pascal_voc(
                images_directory_path=self.val_dir,
                annotations_path=self.val_dir,
                force_masks=self.params.force_masks
                )
            
        return (train_detections, test_detections, val_detections)

    def to_yolo(self):
        super().to_yolo(detections_data=self.detections_data)
    
    def to_coco(self):
        super().to_coco(detections_data=self.detections_data)
    
    def to_jsonl(self):
        super().to_jsonl(detections_data=self.detections_data)


class JsonLFmt(_BaseFmt):
    
    @property
    def detections_data(self):
        train_detections, test_detections, val_detections = (None, None, None)

        # TODO: Need to check for other json file names
        if not hasattr(self, 'dataset_dir'):
            self.logger.error('There needs to be a dataset directory containing the jsonl and image files')
            return (train_detections, test_detections, val_detections)

        train_detections = NewDetectionsDataset.from_jsonl(
            images_directory_path=self.dataset_dir,
            annotations_path=os.path.join(self.dataset_dir, '_annotations.train.jsonl'),
            force_masks=self.params.force_masks
            )

        if self._validate_all_splits():
            test_detections = NewDetectionsDataset.from_jsonl(
                images_directory_path=self.dataset_dir,
                annotations_path=os.path.join(self.dataset_dir, 'instances_test.jsonl'),
                force_masks=self.params.force_masks
                )

            val_detections = NewDetectionsDataset.from_jsonl(
                images_directory_path=self.dataset_dir,
                annotations_path=os.path.join(self.dataset_dir, 'instances_val.jsonl'),
                force_masks=self.params.force_masks
                )
        
        return (train_detections, test_detections, val_detections)

    def to_coco(self):
        return super().to_coco(detections_data=self.detections_data)
    
    def to_pascal(self):
        return super().to_pascal(detections_data=self.detections_data)
    
    def to_yolo(self):
        return super().to_yolo(detections_data=self.detections_data)