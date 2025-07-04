import pytest
import os
import glob
from baseballcv.datasets import DatasetTranslator
from baseballcv.datasets.translation.formats import YOLOFmt, COCOFmt, JsonLFmt, PascalFmt

class TestDatasetTranslator:
    """
    Test suite for the `DatasetTranslator`

    This suite tests the various capabilities of translating different kinds of datasets such
    as coco, yolo, jsonl, and pascal voc. For now, these tests are written to assume the dataset
    formats are in the format you would see on Roboflow.
    """

    @pytest.fixture(scope='module')
    def setup(self, tmp_path_factory) -> dict:
        base = tmp_path_factory.mktemp("conversions")

        conversions = {}
        for conversion in ['yolo', 'coco', 'pascal', 'jsonl', 'dummy']:
            conv_dir = base / conversion
            conv_dir.mkdir()
            conversions[conversion] = conv_dir

        return conversions
    
    def check_yolo(self, pth: str) -> None:
        """
        Checks the YOLO format

        Args:
            pth (str): The input folder directory containing the YOLO files

        This checks for:
        - An existing train and test folder, required for public implemtation
        - An existing yaml file
        - The labeled files end with .tst
        """
        all_pths = os.listdir(pth)
        assert all(name in all_pths for name in ['train', 'test']), 'The paths should contain at least a train and test directory'
        yaml_files = glob.glob(os.path.join(pth, '*.y?ml'))
        assert len(yaml_files) == 1, 'Should only be one yaml file'
        
        # Checks if labels are there
        for subdir in all_pths:
            if subdir in ['train', 'test', 'valid']:
                labels_pth = os.listdir(os.path.join(pth, subdir, 'labels'))
                assert len(labels_pth) > 0, 'There should be labels'
                assert all(os.path.splitext(f)[1] == '.txt' for f in labels_pth), 'The labels should be .txt files'
        
    def check_coco(self, pth: str) -> None:
        """
        Checks the COCO format

        Args:
            pth (str): The input folder directory containing the COCO files

        
        This checks for:
        - An existing train and test folder, required for public implemtation
        - The split files contain a .json file
        """
        all_pths = os.listdir(pth)
        assert all(name in all_pths for name in ['train', 'test']), 'The paths should contain at least a train and test directory'

        for subdir in all_pths:
            if subdir in ['train', 'test', 'valid']:
                assert len(glob.glob(os.path.join(pth, subdir, '*.json'))) == 1, 'There should only be one json file containing annotations'

    def check_pascal_voc(self, pth: str) -> None:
        """
        Checks the PASCAL VOC format

        Args:
            pth (str): The input folder directory containing the PASCAL VOC files

        This checks for:
        - An existing train and test folder, required for public implemtation
        - The split files contain .xml files
        """
        all_pths = os.listdir(pth)
        assert all(name in all_pths for name in ['train', 'test']), 'The paths should contain at least a train and test directory'

        for subdir in all_pths:
            if subdir in ['train', 'test', 'valid']:
                assert len(glob.glob(os.path.join(pth, subdir, '*.xml'))) >= 1, 'There should only be at least one xml file containing annotations'

    def check_jsonl(self, pth: str) -> None:
        """
        Checks the JSONL format

        Args:
            pth (str): The input folder directory containing the JSONL files

        This checks for:
        - An existing dataset folder, required for public implementation
        - The split files are jsonl and there are at least 2 (train, test, optional valid)
        """
        all_pths = os.listdir(pth)
        assert 'dataset' in all_pths, 'The paths should contain at least a dataset directory'

        assert len(glob.glob(os.path.join(pth, 'dataset', '*.jsonl'))) >= 2, 'There should be at least a train and test jsonl file inside.'

    def test_init_(self, setup) -> None:
        """
        Test the initialization of the `DatasetTranslator`

        This makes sure the input formats are properly initialized and
        create the correct instance of each formatter class.
        """
        translator = DatasetTranslator('CoCo', 'YOLO', setup['dummy'], setup['dummy'])
        assert isinstance(translator.fmt_instance, COCOFmt)

        translator = DatasetTranslator('yoLo', 'coco', setup['dummy'], setup['dummy'])
        assert isinstance(translator.fmt_instance, YOLOFmt)

        translator = DatasetTranslator('PasCal', 'YOLO', setup['dummy'], setup['dummy'])
        assert isinstance(translator.fmt_instance, PascalFmt)

        translator = DatasetTranslator('JSONL', 'YOLO', setup['dummy'], setup['dummy'])
        assert isinstance(translator.fmt_instance, JsonLFmt)
    
    def test_invalid_params(self, setup) -> None:
        """
        Tests for failutes of the Translator class

        This checks for:
        - Improper alias names
        - Using Segmentation for jsonl, it's not supported at the moment.
        """
        with pytest.raises(ValueError):
            DatasetTranslator('coco2', 'jsonl', setup['dummy'])
            DatasetTranslator('coco', 'jsonl2', setup['dummy'])
            DatasetTranslator('coco', 'jsonl', setup['dummy'], force_masks=True) # Not supporting this for now

    def test_coco_conversion(self, load_dataset, setup) -> None:
        # jsonl -> coco
        """
        Validates the conversion of a COCO dataset
        This tests the conversion from JSONL -> COCO

        Args:
            load_dataset: Fixture for what formatted dataset to load
            setup: Fixture for what converted dataset to load
        """

        self.check_jsonl(load_dataset['jsonl'])

        DatasetTranslator('jsonl', 'coco', load_dataset['jsonl'], setup['coco']).convert()

        self.check_coco(setup['coco'])
    
    def test_yolo_conversion(self, load_dataset, setup) -> None:
        # coco -> yolo
        """
        Validates the conversion of a YOLO dataset
        This tests the conversion from COCO -> YOLO

        Args:
            load_dataset: Fixture for what formatted dataset to load
            setup: Fixture for what converted dataset to load
        """
        self.check_coco(load_dataset['coco'])

        DatasetTranslator('coco', 'yolo', load_dataset['coco'], setup['yolo']).convert()

        self.check_yolo(setup['yolo'])

    def test_pascal_voc_conversion(self, load_dataset, setup) -> None:
        # yolo -> pascal_voc
        """
        Validates the conversion of a PASCAL VOC dataset
        This tests the conversion from YOLO -> Pascal VOC

        Args:
            load_dataset: Fixture for what formatted dataset to load
            setup: Fixture for what converted dataset to load
        """

        self.check_yolo(load_dataset['yolo'])

        DatasetTranslator('yolo', 'pascal', load_dataset['yolo'], setup['pascal']).convert()

        self.check_pascal_voc(setup['pascal'])

    
    def test_jsonl_conversion(self, load_dataset, setup) -> None:
        # pascal_voc -> jsonl
        """
        Validates the conversion of a JSONL dataset
        This tests the conversion from PASCAL VOC -> JSONL

        Args:
            load_dataset: Fixture for what formatted dataset to load
            setup: Fixture for what converted dataset to load
        """

        self.check_pascal_voc(load_dataset['pascal'])

        DatasetTranslator('pascal', 'jsonl', load_dataset['pascal'], setup['jsonl']).convert()

        self.check_jsonl(setup['jsonl'])
