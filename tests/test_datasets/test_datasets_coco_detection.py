import pytest
import torch
import os
import shutil
from typing import Tuple, Generator, Any
from transformers import DetrImageProcessor
from baseballcv.datasets import CocoDetectionDataset

class TestCocoDetectionDataset:

    @pytest.fixture(scope='module')
    def setup(self, load_dataset) -> Generator[Tuple[str, DetrImageProcessor], Any, None]:
        """Download and set up the actual COCO dataset provided by BaseballCV for testing
        
        This fixture downloads a baseball-specific COCO-formatted dataset and initializes
        an image processor for testing the CocoDetectionDataset class. The fixture handles
        both setup and cleanup of test resources.
        
        Args:
            load_dataset: Fixture providing sample dataset dirs
            
        Returns:
            tuple: A tuple containing:
                - dataset_path (str): Path to the downloaded dataset
                - processor (DetrImageProcessor): Configured image processor for the dataset
        """

        dataset_path = load_dataset['coco']

        # Create dummy dataset to satisfy `CocoDetectionDataset` hierarchical structure (they're all the same),
        # this format is just a little picky.
        expected_pth = os.path.join(dataset_path, 'train', '_coco.json')

        if not os.path.exists(expected_pth):
            shutil.copy(os.path.join(dataset_path, 'train', '_annotations.coco.json'), expected_pth)

        processor = DetrImageProcessor(do_resize=True, size={"height": 800, "width": 800})
        
        yield dataset_path, processor

        os.remove(expected_pth)
    
    def test_coco_dataset_initialization(self, setup, logger) -> None:
        """Test the initialization of CocoDetectionDataset
        
        Verifies that the CocoDetectionDataset initializes correctly with the provided
        dataset path and processor. Checks that essential attributes are properly set
        and that the dataset contains items.
        
        Args:
            setup_coco_dataset: Fixture providing dataset path and processor
            
        Assertions:
            - Image directory path should be set
            - Annotation file path should be set
            - Dataset should contain at least one item
            - Dataset processor should be of the correct type
        """
        dataset_path, processor = setup
        
        split = "train"
        dataset = CocoDetectionDataset(dataset_dir=dataset_path, split=split, processor=processor, logger=logger)
        
        assert dataset.img_dir is not None
        assert dataset.ann_file is not None
        assert len(dataset) > 0
        assert isinstance(dataset.processor, DetrImageProcessor)
    
    def test_coco_dataset_getitem(self, setup, logger) -> None:
        """Test the __getitem__ method of CocoDetectionDataset
        
        Verifies that the dataset's item retrieval functionality works correctly by
        accessing the first item and checking that it returns properly formatted
        tensors and target dictionaries with the expected structure.
        
        Args:
            setup: Fixture providing dataset path and processor
            
        Assertions:
            - Retrieved pixel values should be a PyTorch tensor
            - Target should be a dictionary
            - Target should contain 'class_labels' and 'boxes' keys
            
        """
        dataset_path, processor = setup
        
        split = "train"
        dataset = CocoDetectionDataset(dataset_dir=dataset_path, split=split, processor=processor, logger=logger)
        
        if len(dataset) > 0:
            pixel_values, target = dataset[0]
            
            assert isinstance(pixel_values, torch.Tensor)
            assert 'class_labels' in target
            assert 'boxes' in target