import cv2
import numpy as np
import pytesseract

from text_extract_api.extract.strategies.strategy import Strategy
from text_extract_api.files.file_formats.file_format import FileFormat
from text_extract_api.files.file_formats.image import ImageFileFormat


class TesseractStrategy(Strategy):

    @classmethod
    def name(cls) -> str:
        return "tesseract"

    def extract_text(self, file_format: FileFormat):

        if (
                not isinstance(file_format, ImageFileFormat)
                and not file_format.can_convert_to(ImageFileFormat)
        ):
            raise Exception(f"Tesseract format {file_format.mime_type} is not supported (yet?)")

        images = FileFormat.convert_to(file_format, ImageFileFormat);
        extracted_text = ""

        for i, image in enumerate(images):
            rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            page_text = pytesseract.image_to_string(rgb_image)
            extracted_text += f"--- Page {i + 1} ---\n{page_text}\n"

        return extracted_text
