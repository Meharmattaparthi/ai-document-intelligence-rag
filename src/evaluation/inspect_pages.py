import pymupdf


PDF_PATH = "data/raw/metformin.pdf"


def inspect_pages(page_numbers):
    document = pymupdf.open(PDF_PATH)

    for page_number in page_numbers:
        page = document[page_number - 1]

        print("\n" + "=" * 80)
        print(f"PAGE {page_number}")
        print("=" * 80)
        print(page.get_text())

    document.close()


if __name__ == "__main__":
    inspect_pages([1, 4, 5, 8, 11, 12, 16])