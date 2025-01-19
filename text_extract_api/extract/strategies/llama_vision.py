import contextlib
import os
import tempfile
import time
from typing import Optional

import ollama

from text_extract_api.extract.strategies.strategy import Strategy
from text_extract_api.files.file_formats.file_format import FileFormat
from text_extract_api.files.file_formats.image import ImageFileFormat


class LlamaVisionStrategy(Strategy):
    """Llama 3.2 Vision OCR Strategy"""

    @classmethod
    def name(cls) -> str:
        return "llama_vision"

    def extract_text(self, file_format: FileFormat, language: str = 'en') -> str:

        if (
                not isinstance(file_format, ImageFileFormat)
                and not file_format.can_convert_to(ImageFileFormat)
        ):
            raise TypeError(
                f"Llama Vision - format {file_format.mime_type} is not supported (yet?)"
            )

        images = FileFormat.convert_to(file_format, ImageFileFormat)
        start_time = time.time()
        total_pages = len(images)

        extracted_texts = []
        for i, image in enumerate(images, 1):
            text = self.process_single_image(
                image=image,
                start_time=start_time,
                page_num=i,
                total_pages=total_pages
            )
            if text:
                extracted_texts.append(text)

        return "\n\n".join(extracted_texts)


    @contextlib.contextmanager
    def create_temp_image(self, image_binary: bytes) -> str:
        """Context manager to handle temporary file creation and cleanup"""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_file.write(image_binary)
            temp_file.close()
            yield temp_file.name
        finally:
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass  # @todo send to logger after its implementation

    def process_single_image(self, image: FileFormat, start_time: float,
                             page_num: int, total_pages: int) -> Optional[str]:
        """Process a single image and return extracted text"""
        with self.create_temp_image(image.binary) as temp_filename:
            try:
                response = ollama.chat("llama3.2-vision", [{
                    'role': 'user',
                    'content': os.getenv('LLAMA_VISION_PROMPT', "You are OCR. Convert image to markdown."), # @todo don't get this directly from env
                    'images': [temp_filename]
                }], stream=True)

                extracted_text = ""
                for chunk_num, chunk in enumerate(response, 1):
                    meta = {
                        'progress': str(30 + int(20 / total_pages)),
                        'status': (f'OCR Processing (page {page_num} of {total_pages}) '
                                   f'chunk no: {chunk_num}'),
                        'start_time': start_time,
                        'elapsed_time': time.time() - start_time
                    }
                    self.update_state_callback(state='PROGRESS', meta=meta)
                    extracted_text += chunk['message']['content']

                return extracted_text
            except ollama.ResponseError as e:
                print('Error:', e.error)
                raise Exception(f"Failed to process page {page_num} with Llama Vision: {str(e)}")
