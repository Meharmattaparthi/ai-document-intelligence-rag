import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to a PDF file.

    Returns:
        list: Extracted text for each page.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/raw/metformin.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")

    print("\nFirst page:")
    print(pages[0]["text"][:2000])