import pytest
import multiprocessing as mp
from unittest.mock import Mock
import requests
from baseballcv.functions import LoadTools, BaseballTools
from baseballcv.utilities import BaseballCVLogger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from unittest import mock
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing() -> None:
    """
    Ensures that the multiprocessing start method is set to 'spawn' for tests.
    
    This fixture runs automatically once per test session and configures the
    multiprocessing start method to 'spawn' which is more compatible with
    pytest and avoids potential issues with forking processes during testing.
    
    Returns:
        None: This fixture doesn't return any value.
    """
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    return None

@pytest.fixture
def load_tools() -> LoadTools:
    """
    Provides a LoadTools instance for testing.
    
    Creates and returns a LoadTools object that can be used in tests to load
    datasets, models, and other resources needed for testing the baseballcv
    package functionality.

    Returns:
        LoadTools: An instance of LoadTools.
    """
    return LoadTools()

@pytest.fixture(scope='session') # Only run once
def load_dataset() -> Dict[str, str]:
    """
    Returns the respective dataset path for each tested dataset type. 
    Use this for any tests on datasets, not datasets in the API due to 
    their size, which can slow down testing speed.

    Returns:
        Dict[str, str]: An iterable of each dataset stored in a dictionary.
    """
    return {
        'coco': 'tests/data/test_datasets/coco_stuff',
        'jsonl': 'tests/data/test_datasets/jsonl_stuff',
        'pascal': 'tests/data/test_datasets/pascal_stuff',
        'yolo': 'tests/data/test_datasets/yolo_stuff'
    }

@pytest.fixture
def baseball_tools() -> BaseballTools:
    """
    Provides a BaseballTools instance for testing.
    
    Creates and returns a BaseballTools object that can be used in tests
    to verify the functionality of baseball-specific data processing and
    analysis tools provided by the baseballcv package.

    Returns:
        BaseballTools: An instance of BaseballTools.
    """
    return BaseballTools()

@pytest.fixture
def logger() -> BaseballCVLogger:
    """
    Creates and returns a BaseballCVLogger instance that can be used in tests
    to verify the functionality of logging and logging messages.

    Returns:
        BaseballCVLogger: An instance of BaseballCVLogger.
    """
    return BaseballCVLogger.get_logger("TestLogger")

@pytest.fixture
def mock_responses() -> tuple:
    """
    Provides mock HTTP responses for testing network requests.
    
    Creates and returns two mock response objects:
    1. A success response (200) with mock file content and headers
    2. An error response (404) that raises an HTTPError when raise_for_status is called
    
    These mock responses can be used to test functions that make HTTP requests
    without actually connecting to external services.

    Returns:
        tuple: A tuple containing (success_response, error_response) mock objects.
    """
    success = Mock()
    success.status_code = 200
    success.content = b"mock file content"
    success.headers = {"Content-Disposition": "attachment; filename=model.pt"}
    success.raise_for_status.return_value = None  

    # Create error response
    error = Mock()
    error.status_code = 404
    error.json.return_value = {"error": "File not found"}
    http_error = requests.exceptions.HTTPError("404 Client Error: Not Found")
    http_error.response = error  
    error.raise_for_status.side_effect = http_error  

    return success, error

@pytest.fixture
def mock_model() -> Mock:
    """
    Provides a mock model for testing.
    
    Creates and returns a mock model object that can be used in tests to
    verify the functionality of model training and evaluation.

    Returns:
        Mock: A mock model object.
    """
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = mock.MagicMock()
            self.vision_model.encoder = mock.MagicMock()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, pixel_values):
            return {"logits": torch.rand(1, 2)}
        
    return MockModel()

@pytest.fixture
def reset_logger_registry():
    """
    Reset the BaseballCVLogger registry before and after each test.
    
    This fixture ensures that each test starts with a clean logger registry
    and restores the original registry after the test completes.
    
    Yields:
        None
    """
    original_loggers = BaseballCVLogger._loggers.copy()
    BaseballCVLogger._loggers = {}
    yield
    BaseballCVLogger._loggers = original_loggers


# Network testing configurations
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )

def pytest_runtest_setup(item):
    if "network" in item.keywords and os.environ.get("SKIP_NETWORK_TESTS", "0") == "1":
        pytest.skip("Network tests disabled")