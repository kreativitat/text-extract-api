from marker.convert import convert_single_pdf
from marker.models import load_all_models
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF/Office/Image file.")
    parser.add_argument("file", type=str, nargs='?', default="./examples/example-mri.pdf", help="The path to the PDF/Office/Image file to be processed.")
    args = parser.parse_args()

    model_lst = load_all_models()
    pdf_file_path = args.file
    full_text, images, out_meta = convert_single_pdf(pdf_file_path, model_lst)
    print(full_text)

