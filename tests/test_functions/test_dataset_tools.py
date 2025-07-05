import pytest
import os
import tempfile
from baseballcv.functions.dataset_tools import DataTools
from baseballcv.functions.load_tools import LoadTools

class TestDatasetTools:
    """ Test class for various Dataset Generation Tools """

    @pytest.fixture(scope='module') # Only run this once
    def setup(self, tmp_path_factory) -> str:
        """ Sets up the environment for Dataset Tools"""
    
        tmp_dir = tmp_path_factory.mktemp("dataset_tools")
        temp_dir = tmp_dir / "frames"
        temp_video_dir = tmp_dir / "videos"
        temp_dir.mkdir()
        temp_video_dir.mkdir()

        DataTools().generate_photo_dataset(
            output_frames_folder=temp_dir,
            video_download_folder=temp_video_dir,
            max_plays=2,
            max_num_frames=10,
            use_savant_scraper=False,
            input_video_folder='tests/data/test_functions/savant_scraper_ex_videos'
        )

        return str(temp_dir)

    def test_generate_photo_dataset(self, setup):
        """
        Tests the generate_photo_dataset method using example call.
        Keeps part of output for the following test_automated_annotation method.
        """
        temp_dir = setup

        assert os.path.exists(temp_dir)
        frames = os.listdir(temp_dir)
        assert len(frames) > 0, "Should generate some frames"

        for frame in frames:
            assert frame.endswith(('.jpg', '.png')), f'Invalid frame format {frame}'

    @pytest.mark.parametrize("mode", ["autodistill", "legacy"])
    def test_automated_annotation(self, setup, mode):
        """
        Tests the annotation tools to make sure the proper file systems are loaded 
        and manipulated.
        """
        temp_dir = setup

        ontology = { "a mitt worn by a baseball player for catching a baseball": "glove" } if mode == "autodistill" else None

        legacy_annotation_dir = tempfile.mkdtemp()
        autodistill_annotation_dir = tempfile.mkdtemp()
        annotation_dir = legacy_annotation_dir if mode != "autodistill" else autodistill_annotation_dir

        with tempfile.TemporaryDirectory() as annotation_dir:
            DataTools().automated_annotation(
                model_alias="glove_tracking",
                model_type="detection",
                image_dir=temp_dir,
                output_dir=annotation_dir if mode != "autodistill" else f"{annotation_dir}_autodistill",
                conf=0.5,
                mode=mode,
                ontology=ontology,
                batch_size=100
            )
            assert os.path.exists(annotation_dir)

            if mode != "autodistill":
                assert os.path.exists(os.path.join(annotation_dir, "annotations"))
                images = os.listdir(annotation_dir)
                annotations = os.listdir(os.path.join(annotation_dir, "annotations"))
                assert len(images) > 0, "Should have copied some images"
                assert len(annotations) > 0, "Should have generated some annotations"
                
                for ann_file in annotations:
                    assert ann_file.endswith('.txt'), f"Invalid annotation format: {ann_file}"
                
                os.remove(LoadTools().yolo_model_aliases['glove_tracking'].replace('.txt', '.pt')) # Remove the loaded pytorch file

            else:
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "train", "labels")), "Should have labels"
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "train", "images")), "Should have images"
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "data.yaml")), "Should have data.yaml"
                images = os.listdir(os.path.join(f"{annotation_dir}_autodistill", "train", "images"))
                annotations = os.listdir(os.path.join(f"{annotation_dir}_autodistill", "train", "labels"))
                assert len(images) > 0, "Should have copied some images"
                assert len(annotations) > 0, "Should have generated some annotations"
