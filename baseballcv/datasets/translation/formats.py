from supervision import DetectionDataset
import os
import shutil
import random
from typing import Tuple, List, Optional
import math # For floating point rounding issues
from baseballcv.utilities import BaseballCVLogger
import glob

class _BaseFmt:

# TODO: Might need to add functionality for the json data?
    def __init__(self, params):
        self.params = params
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

        dir_list = os.listdir(self.params.root_dir)


        train_dir = self._find_respective_file(dir_list, 'train')
        test_dir = self._find_respective_file(dir_list, 'test')
        val_dir = self._find_respective_file(dir_list, 'val')
        annotations_dir = self._find_respective_file(dir_list, 'annotation')

        if train_dir is None:
            raise FileNotFoundError('There needs to at least be a `train` folder')

        dir_attrs = {
            "train_dir": train_dir,
            "test_dir": test_dir,
            "val_dir": val_dir,
            "annotations_dir": annotations_dir,
        }

        for attr_name, dir_name in dir_attrs.items():
            if dir_name is not None:
                full_path = os.path.join(self.params.root_dir, dir_name)
                setattr(self, attr_name, full_path)

        self.new_dir = f"{self.params.root_dir}_conversion"
        os.makedirs(self.new_dir, exist_ok=True)


    @property
    def detections_data(self): raise NotImplementedError

    def to_coco(self, detections_data: Tuple):
        train_det, test_det, val_det = detections_data
        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'train'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_train.json'))
        
        if test_det and val_det:
            test_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'test'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_test.json'))
            val_det.as_coco(images_directory_path=os.path.join(self.new_dir, 'val'), 
                          annotations_path=os.path.join(self.new_dir, 'annotations', 'instances_val.json'))
            
    def to_yolo(self, detections_data: Tuple):
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

    def to_pascal(self, detections_data: Tuple): 
        train_det, test_det, val_det = detections_data

        class_labels = train_det.classes # Will be used to stratify training splits

        train_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'train', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'train', 'labels'))
        
        if test_det and val_det:
            test_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'test', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'test', 'labels'))
            val_det.as_pascal_voc(images_directory_path=os.path.join(self.new_dir, 'val', 'images'), 
                          annotations_directory_path=os.path.join(self.new_dir, 'val', 'labels'))


    def move_to_new_folder(self, original_dir, new_dir, *, random_state=None, shuffle=True, stratify: str=None):
        if self.params.train_test_val_split and not self._validate_all_splits():
            try:
                train_items = os.listdir(f"{self.train_dir}/images")
                # For now, move the images, will need to figure out how to move annotations
                train, test, val = self._train_test_val_split(data=train_items, split=self.params.train_test_val_split, 
                                                              random_state=random_state, 
                                                              shuffle=shuffle, 
                                                              stratify=stratify)

                new_subdirs = {
                    os.path.basename(self.train_dir): train,
                    'test': test,
                    'val': val
                }
                
                # May need to make this bit faster
                for subdir_name, file_list in new_subdirs.items():
                    target = os.path.join(self.new_dir, subdir_name, 'images')
                    os.makedirs(target, exist_ok=True)

                    for pth in file_list:
                        shutil.copy(os.path.join(f"{self.train_dir}/images", pth), target)
                
                self.logger.info(f"Directory Copied over to {self.new_dir}")

            except Exception as e:
                self.logger.exception(f"Issue with splitting the train folder. Please check to make sure the folder contains a `images` + `labels` subdirectory. {e}")
        
        else:
            # Copy over all files to the conversion folder
            # TODO: Check to see if this is efficient
            shutil.copytree(self.params.root_dir, self.new_dir, dirs_exist_ok=True)
            self.logger.info(f"Directory Copied over to {self.new_dir}")


    def _find_respective_file(self, dir_list: List[str], query: str) -> Optional[str]:
        for item in dir_list:
            if query in item:
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
        try:
            yaml_pth = glob.glob(os.path.join(self.params.root_dir, '**', '*.y?ml'), recursive=True)[0]
        except IndexError:
            self.logger.error('Make sure you have a specified yaml file in your directory.')

        train_detections = DetectionDataset.from_yolo(
            images_directory_path=os.path.join(self.params.root_dir, self.train_dir, 'images'),
            annotations_directory_path=os.path.join(self.params.root_dir, self.train_dir, 'labels'),
            data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
            )

        if self._validate_all_splits():
            test_detections = DetectionDataset.from_yolo(
                images_directory_path=os.path.join(self.params.root_dir, self.test_dir, 'images'),
                annotations_directory_path=os.path.join(self.params.root_dir, self.test_dir, 'labels'),
                data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
                )

            val_detections = DetectionDataset.from_yolo(
                images_directory_path=os.path.join(self.params.root_dir, self.val_dir, 'images'),
                annotations_directory_path=os.path.join(self.params.root_dir, self.val_dir, 'labels'),
                data_yaml_path=yaml_pth, force_masks=self.params.force_masks, is_obb=self.params.is_obb
                )
        return (train_detections, test_detections, val_detections)
        
    def to_coco(self):
        super().to_coco(detections_data=self.detections_data)

    def to_pascal(self):
        super().to_pascal(detections_data=self.detections_data)
        

class COCOFmt(_BaseFmt):
    # Requires self.annotations_file so if not exists, raise valueerror
    @property
    def detections_data(self):
        # TODO: Need to check for other json file names
        if not hasattr(self, 'annotations_file'):
            self.logger.error('There needs to at least be an annotation file')
            return (None, None, None)

        train_detections = DetectionDataset.from_coco(
            images_directory_path=os.path.join(self.params.root_dir, self.train_dir, 'images'),
            annotations_path=os.path.join(self.params.root_dir, self.annotations_file, 'instances_train.json'),
            force_masks=self.params.force_masks
            )

        if self._validate_all_splits():
            test_detections = DetectionDataset.from_coco(
                images_directory_path=os.path.join(self.params.root_dir, self.test_dir, 'images'),
                annotations_path=os.path.join(self.params.root_dir, self.annotations_file, 'instances_test.json'),
                force_masks=self.params.force_masks
                )

            val_detections = DetectionDataset.from_coco(
                images_directory_path=os.path.join(self.params.root_dir, self.val_dir, 'images'),
                annotations_path=os.path.join(self.params.root_dir, self.annotations_file, 'instances_val.json'),
                force_masks=self.params.force_masks
                )
        return (train_detections, test_detections, val_detections)
        
    def to_yolo(self):
        super().to_yolo(detections_data=self.detections_data)

    def to_pascal(self):
        super().to_pascal(detections_data=self.detections_data)
        
class PascalFmt(_BaseFmt):
    @property
    def detections_data(self):
        if not hasattr(self, 'annotations_file'):
            self.logger.error('There needs to at least be an annotation file')
            return (None, None, None)

        train_detections = DetectionDataset.from_pascal_voc(
            images_directory_path=os.path.join(self.params.root_dir, self.train_dir, 'images'),
            annotations_path=os.path.join(self.params.root_dir, self.train_dir, 'labels'),
            force_masks=self.params.force_masks
            )

        if self._validate_all_splits():
            test_detections = DetectionDataset.from_pascal_voc(
                images_directory_path=os.path.join(self.params.root_dir, self.test_dir, 'images'),
                annotations_path=os.path.join(self.params.root_dir, self.test_dir, 'labels'),
                force_masks=self.params.force_masks
                )

            val_detections = DetectionDataset.from_pascal_voc(
                images_directory_path=os.path.join(self.params.root_dir, self.val_dir, 'images'),
                annotations_path=os.path.join(self.params.root_dir, self.val_dir, 'labels'),
                force_masks=self.params.force_masks
                )
        return (train_detections, test_detections, val_detections)

    def to_yolo(self):
        super().to_yolo(detections_data=self.detections_data)
    
    def to_coco(self):
        super().to_coco(detections_data=self.detections_data)