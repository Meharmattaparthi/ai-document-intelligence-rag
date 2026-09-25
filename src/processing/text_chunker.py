import re
import json
import sys
import os


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def detect_section(text):
    """Detect the most relevant section heading in a text block."""

    lines = text.splitlines()

    important_sections = [
        "CONTRAINDICATIONS",
        "WARNINGS AND PRECAUTIONS",
        "WARNINGS",
        "DOSAGE AND ADMINISTRATION",
        "DRUG INTERACTIONS",
        "ADVERSE REACTIONS",
        "USE IN SPECIFIC POPULATIONS",
        "DESCRIPTION",
        "INDICATIONS AND USAGE",
        "OVERDOSAGE",
        "CLINICAL PHARMACOLOGY",
        "CLINICAL STUDIES",
    ]

    for line in lines:
        cleaned = line.strip()

        if not cleaned:
            continue

        upper = cleaned.upper()

        # Prefer an exact heading match.
        for section in important_sections:
            if upper == section:
                return section

    return "GENERAL"


def create_chunks(
    pages,
    document_name="document.pdf",
    chunk_size=1500,
    overlap=200
):
    chunks = []

    important_sections = [
        "CONTRAINDICATIONS",
        "WARNINGS AND PRECAUTIONS",
        "WARNINGS",
        "DOSAGE AND ADMINISTRATION",
        "DOSAGE FORMS AND STRENGTHS",
        "DRUG INTERACTIONS",
        "ADVERSE REACTIONS",
        "USE IN SPECIFIC POPULATIONS",
        "DESCRIPTION",
        "INDICATIONS AND USAGE",
        "OVERDOSAGE",
        "CLINICAL PHARMACOLOGY",
        "CLINICAL STUDIES",
    ]

    for page in pages:
        cleaned_text = clean_text(page["text"])
        lines = cleaned_text.splitlines()

        current_section = "GENERAL"
        current_text = ""

        for line in lines:
            line = line.strip()

            if not line:
                continue

            upper_line = line.upper()

            detected_section = None

            for section in important_sections:
                if upper_line == section:
                    detected_section = section
                    break

            if detected_section:
                if current_text.strip():
                    chunks.append({
                        "document": document_name,
                        "page_number": page["page_number"],
                        "section": current_section,
                        "text": current_text.strip()
                    })

                current_section = detected_section
                current_text = line

                continue

            if current_text:
                current_text += "\n" + line
            else:
                current_text = line

            if len(current_text) >= chunk_size:
                chunks.append({
                    "document": document_name,
                    "page_number": page["page_number"],
                    "section": current_section,
                    "text": current_text.strip()
                })

                overlap_text = current_text[-overlap:]
                current_text = overlap_text

        if current_text.strip():
            chunks.append({
                "document": document_name,
                "page_number": page["page_number"],
                "section": current_section,
                "text": current_text.strip()
            })

    return chunks


if __name__ == "__main__":

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    sys.path.insert(0, project_root)

    from src.ingestion.document_loader import extract_text_from_pdf

    pdf_path = "data/raw/metformin.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(
        pages,
        document_name="metformin.pdf"
    )

    output_path = "data/processed/metformin_chunks.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Chunks saved to: {output_path}"
    )

    print(
        f"Total pages: {len(pages)}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    print("\nFirst chunk:")
    print(chunks[0]["text"])

    print("\n--- Metadata ---")

    print(
        f"Document: "
        f"{chunks[0]['document']}"
    )

    print(
        f"Page: "
        f"{chunks[0]['page_number']}"
    )

    print(
        f"Section: "
        f"{chunks[0]['section']}"
    )