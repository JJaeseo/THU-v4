"""
Parse PDF or plain-text documents into a single text string.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_document(file_path: str) -> str:
    """Return the full text content of a .txt or .pdf file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\n" not in text and "\\n" in text:
            text = text.replace("\\n", "\n")
            logger.info("Decoded literal \\n sequences to real newlines.")
        logger.info("Loaded TXT: %d chars, %d lines", len(text), text.count("\n"))
        return text
    if ext == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .txt or .pdf.")


def _parse_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path))
        if text and text.strip():
            logger.info("PDF parsed with pdfminer.six: %d chars", len(text))
            return text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pdfminer.six failed: %s", exc)

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages  = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                pages.append(f"[Page {i + 1}]\n{t}")
        if pages:
            text = "\n\n".join(pages)
            logger.info("PDF parsed with pypdf: %d pages, %d chars", len(pages), len(text))
            return text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pypdf failed: %s", exc)

    raise RuntimeError(
        "No PDF parser available.\n  pip install pdfminer.six\nor\n  pip install pypdf"
    )
