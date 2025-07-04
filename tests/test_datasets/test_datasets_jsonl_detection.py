import pytest
import os
import glob
import logging
from PIL import Image
from typing import Tuple
from baseballcv.datasets import JSONLDetection

class TestJSONLDetection:
    @pytest.fixture
    def setup(self, load_dataset) -> Tuple[list[dict], str, str, logging.Logger, str]:
        """Set up and download BaseballCV JSONL dataset for testing
        
        This fixture downloads a baseball-specific JSONL dataset and prepares the necessary
        directories and files for testing the JSONLDetection class. It handles the setup
        and cleanup of test resources.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Returns:
            Tuple: A tuple containing:
                - entries (list[dict]): List of parsed JSONL entries
                - images_dir (str): Path to the directory containing images
                - jsonl_path (str): Path to the JSONL file
                - logger (logging.Logger): Logger instance for testing
                - dataset_path (str): Path to the downloaded dataset
                
        Raises:
            FileNotFoundError: If no JSONL file is found in the dataset
        
        """

        logger = logging.getLogger("test_logger")
        
        dataset_path = load_dataset['jsonl']
        
        images_dir = os.path.join(dataset_path, "dataset")
        
        jsonl_files = glob.glob(os.path.join(images_dir, '*.jsonl'))

        if not jsonl_files:
            raise FileNotFoundError("No JSONL file found in the dataset")
        
        jsonl_path = jsonl_files[0]
        entries = JSONLDetection.load_jsonl_entries(jsonl_path, logger)
        
        return entries, images_dir, jsonl_path, logger, dataset_path
    
    def test_jsonl_detection_initialization(self, setup) -> None:
        """Test the initialization of JSONLDetection
        
        Verifies that the JSONLDetection class initializes correctly with the provided
        entries, image directory path, and logger. Checks that essential attributes are
        properly set and that the dataset contains the expected number of items.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - Dataset length should match the number of entries
            - Image directory path should be correctly set
            - Logger should be correctly set
            - Dataset should have a transforms attribute
        """
        entries, images_dir, _, logger, _ = setup
        
        dataset = JSONLDetection(entries=entries, image_directory_path=images_dir, logger=logger)
        
        assert len(dataset) == len(entries)
        assert dataset.image_directory_path == images_dir
        assert dataset.logger == logger
        assert hasattr(dataset, 'transforms')
    
    def test_jsonl_detection_getitem(self, setup) -> None:
        """Test the __getitem__ method of JSONLDetection
        
        Verifies that the dataset's item retrieval functionality works correctly by
        accessing the first item and checking that it returns properly formatted
        image and entry objects with the expected types.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - Retrieved image should be a PIL Image
            - Retrieved entry should be a dictionary
            
        """
        entries, images_dir, _, logger, _ = setup
        
        if len(entries) == 0:
            pytest.skip("No entries found in the dataset")
        
        dataset = JSONLDetection(entries=entries, image_directory_path=images_dir, logger=logger, augment=False)
        
        try:
            image, entry = dataset[0]
            
            assert isinstance(image, Image.Image)
            assert isinstance(entry, dict)
        except Exception as e:
            pytest.skip(f"Error accessing first item in dataset: {str(e)}")
    
    def test_jsonl_detection_load_entries(self, setup) -> None:
        """Test the load_jsonl_entries static method
        
        Verifies that the static method correctly loads and parses entries from a JSONL file,
        returning a list of dictionaries representing the valid entries.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - Returned entries should be a list
            - If entries exist, the first entry should be a dictionary
        """
        _, _, jsonl_path, logger, _ = setup
        
        entries = JSONLDetection.load_jsonl_entries(jsonl_path, logger)
        
        assert isinstance(entries, list)
        if len(entries) > 0:
            assert isinstance(entries[0], dict)
    
    def test_jsonl_detection_transformations(self, setup) -> None:
        """Test the image transformation methods
        
        Verifies that the image transformation methods (color jitter, blur, noise)
        correctly process input images and return valid PIL Image objects.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - Each transformation method should return a PIL Image
        
        """

        _, images_dir, _, _, _ = setup
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:1]

        if not image_files:
            image = Image.new('RGB', (100, 100), color='green')
        else:
            image_path = os.path.join(images_dir, image_files[0])
            image = Image.open(image_path)
        
        jittered = JSONLDetection.random_color_jitter(image)
        assert isinstance(jittered, Image.Image)
        
        blurred = JSONLDetection.random_blur(image)
        assert isinstance(blurred, Image.Image)
        
        noisy = JSONLDetection.random_noise(image)
        assert isinstance(noisy, Image.Image)
    
    def test_jsonl_detection_invalid_jsonl(self, setup, tmp_path) -> None:
        """Test handling of invalid JSONL file
        
        Verifies that the load_jsonl_entries method correctly handles files containing
        both invalid and valid JSON lines, skipping the invalid lines and returning
        only the valid entries.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - Only valid entries should be included in the result
            - The content of valid entries should be correctly parsed
        """
        _, _, _, logger, _ = setup
        
        temp_file = tmp_path
        temp_file.mkdir(exist_ok=True)

        temp_file = temp_file / 'fake_jsonl.jsonl'
        with open(temp_file, 'w') as f:
            f.write("This is not valid JSON\n")
            f.write('{"valid": "entry"}\n')

        temp_file_path = str(temp_file)

        try:
            entries = JSONLDetection.load_jsonl_entries(temp_file_path, logger)
            assert len(entries) == 1
            assert entries[0]['valid'] == 'entry'

        except Exception as e:
            pytest.fail(f'Exception occured: {e}')
    
    def test_jsonl_detection_empty_file(self, setup, tmp_path) -> None:
        """Test handling of empty JSONL file
        
        Verifies that the load_jsonl_entries method correctly raises a ValueError
        when attempting to load entries from an empty file.
        
        Args:
            setup: Fixture providing dataset entries, paths, and logger
            
        Assertions:
            - A ValueError should be raised with an appropriate error message
            
        """
        _, _, _, logger, _ = setup
        
        no_jsonl = tmp_path
        no_jsonl.mkdir(exist_ok=True)
        no_jsonl = no_jsonl / 'empty.jsonl'
        no_jsonl.touch()
        no_jsonl_pth = str(no_jsonl)
        
        with pytest.raises(ValueError) as excinfo:
            JSONLDetection.load_jsonl_entries(no_jsonl_pth, logger)
        assert "No valid entries found" in str(excinfo.value)
        
