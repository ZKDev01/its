import os 
import aspose.pdf as ap 
import aspose.pydrawing as drawing
from docling.document_converter import DocumentConverter

from src.utils import catch_errors


@catch_errors
def convert_to_md_using_docling(source:str,path_to_export:str) -> None:
  ""
  converter = DocumentConverter()
  result = converter.convert(source=source)
  with open(path_to_export, 'w', encoding='utf-8') as file:
    file.write(result.document.export_to_markdown())


from langchain_community.document_loaders import PyPDFLoader

