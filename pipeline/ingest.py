import re
from pathlib import Path

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

class Ingestor:
    """
    Performs OCR on PDFs and saves the raw text.

    Attributes:
        pipeline: The docling pipeline object to conduct OCR.
        doc_converter: The docling document converter.
    """
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = False
        self.pipeline_options.table_structure_options.do_cell_matching = False
        self.pipeline_options.ocr_options.lang = ["en"]
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads = 8,
            device      = AcceleratorDevice.CUDA
        )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options,
                                                 backend=DoclingParseV2DocumentBackend)
            }
        )

    def clean_text(self, text):
        """
        Removes page/paragraph labels like “2-33.” or “12-4.  ”
        and collapses multiple empty lines.

        :param text: The text to clean.
        :return: The cleaned text.
        """
        strip_page_header_labels = re.compile(r"\b\d+-\d+\.\s*")
        txt = strip_page_header_labels.sub("", text)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    def save_text(self, pdf_path: Path, text_path: Path):
        """
        Executes the scan, clean, and save actions.

        :param pdf_path: PDF to scan.
        :param text_path: Location so save raw text.
        :return: None.
        """
        converted_pdf = self.doc_converter.convert(pdf_path)
        dirty = converted_pdf.document.export_to_text()
        clean = self.clean_text(dirty)
        with text_path.open("w", encoding="utf-8") as fp:
            fp.write(clean)
