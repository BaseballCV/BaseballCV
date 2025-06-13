from typing import Dict, Tuple, Union
from .formats import YOLOFmt, COCOFmt, PascalFmt
from dataclasses import dataclass
import shutil

TRANSLATOR = {
    'yolo': YOLOFmt,
    'pascal': PascalFmt,
    'coco': COCOFmt

}

@dataclass
class ConversionParams:
    root_dir: str
    train_test_val_split: Tuple[float, float, float] = () # optional param if the train, test, val split is not established
    train_test_val_args: Dict[str, Union[str, bool, int]] = None # optional param to specify the train, test, val split.
    force_masks: bool = False
    is_obb: bool = False
    remove_original_dir: bool = False

class DatasetTranslator:
    def __init__(self, format_name: str, conversion_name: str, params: ConversionParams):
        format_name = format_name.lower()
        conversion_name = conversion_name.lower()

        if format_name not in TRANSLATOR:
            raise ValueError(f"Invalid format: {format_name}",
                             f"These are the supported formats: {', '.join(list(TRANSLATOR.keys()))}")
        
        self.fmt_instance = TRANSLATOR[format_name](params)

        if not hasattr(self.fmt_instance, f'to_{conversion_name}'):
            raise ValueError(f'Unsupported format: {format_name} -> {conversion_name}')
        
        self._convert_fn = getattr(self.fmt_instance, f'to_{conversion_name}')

    def convert(self) -> None:
        self._convert_fn()

    @classmethod
    def delete_dir(self, dir_name: str) -> None:
        shutil.rmtree(dir_name)

    def rename(self):
        """Gives user ability to rename files or directories
        """
        pass
    def move(self):
        """Gives user ability to move file from one to another
        """
        pass