import pytest
import os
import json
import logging
from typing import Tuple
from baseballcv.datasets import DataProcessor

@pytest.mark.skip(reason="Appears outdated with the new Translator class. Class needs to be refined anyways")
class TestDataProcessor:
    @pytest.fixture
    def setup_data_processor_and_dataset(self, load_dataset) -> Tuple[DataProcessor, str, dict]:
        """Set up a DataProcessor instance and temporary directories
        
        This fixture initializes a DataProcessor instance with a logger and loads a baseball
        dataset for testing. It provides the processor, dataset path, and class dictionary
        for use in tests, and handles cleanup after tests complete.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Returns:
            tuple: A tuple containing:
                - processor (DataProcessor): Initialized DataProcessor instance
                - processed_dataset_path (str): Path to the downloaded dataset
                - dict_classes (dict): Dictionary mapping class IDs to class names
                
        """
        logger = logging.getLogger("test_logger")
        
        processor = DataProcessor(logger)
        dict_classes = {0: 'batter', 1: 'pitcher'}

        processed_dataset_path = load_dataset['yolo']
        
        return processor, processed_dataset_path, dict_classes

    def test_prepare_dataset_with_existing_split(self, setup_data_processor_and_dataset) -> None:
        """Test prepare_dataset method with an existing train/test/valid split
        
        Verifies that the prepare_dataset method correctly processes a dataset that
        already has train/test/valid splits. Checks that the method returns the expected
        number of paths and that the directory structure is created correctly.
        
        Args:
            setup_data_processor_and_dataset: Fixture providing processor, dataset path, and class dictionary
            
        Assertions:
            - Result should contain 5 paths
            - Directory structure should include images and labels folders for each split
        """
        processor, processed_dataset_path, dict_classes = setup_data_processor_and_dataset
        
        result = processor.prepare_dataset(processed_dataset_path, dict_classes)
        
        assert len(result) == 5, "Should return 5 paths"
        
        for split in ["train", "test", "valid"]:
            assert os.path.exists(os.path.join(processed_dataset_path, split, "images"))
            assert os.path.exists(os.path.join(processed_dataset_path, split, "labels"))
    
    def test_convert_annotations(self, setup_data_processor_and_dataset) -> None:
        """Test convert_annotations method
        
        Verifies that the convert_annotations method correctly converts dataset annotations
        to the required format. Checks that the output file exists and contains properly
        formatted annotations with the expected keys and values.
        
        Args:
            setup_data_processor_and_dataset: Fixture providing processor, dataset path, and class dictionary
            
        Assertions:
            - Output file should exist
            - Annotations should contain required keys (image, prefix, suffix)
            - Prefix should be set to "<OD>"
            - Suffix should contain the expected class name
        """
        processor, processed_dataset_path, dict_classes = setup_data_processor_and_dataset
        split = "train"
        
        output_file = processor.convert_annotations(processed_dataset_path, split, dict_classes)
        
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            first_annotation = json.loads(lines[0])
            assert "image" in first_annotation, "image key should be present"
            assert "prefix" in first_annotation, "prefix key should be present"
            assert "suffix" in first_annotation, "suffix key should be present"
            assert first_annotation["prefix"] == "<OD>", "prefix should be <OD>"
            assert "baseball<loc_" in first_annotation["suffix"], "suffix should contain baseball class"
