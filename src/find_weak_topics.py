
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "uploads"

topics = {
    "Q24 - MySQL data types": [
        "VARCHAR", "INT", "INTEGER", "DECIMAL",
        "FLOAT", "DOUBLE", "DATE", "DATETIME",
        "TEXT", "BOOLEAN", "BLOB"
    ],
    "Q27 - For and while loops": [
        "for loop", "while loop", "for statement",
        "while statement", "range("
    ],
    "Q28 - Break, continue and pass": [
        "break statement", "continue statement",
        "pass statement", "break", "continue", "pass"
    ]
}

for pdf_path in PDF_DIR.rglob("*.pdf"):
    print(f"\n{'=' * 65}")
    print("PDF:", pdf_path.name)
    print("=" * 65)

    try:
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            lower_text = text.lower()

            for topic, keywords in topics.items():
                matches = [
                    word for word in keywords
                    if word.lower() in lower_text
                ]

                if len(matches) >= 2:
                    print(f"\n{topic}")
                    print(f"Page: {page_number}")
                    print("Matched terms:", ", ".join(matches))
                    print("Text:", " ".join(text.split())[:450])

    except Exception as error:
        print(f"Could not read {pdf_path.name}: {error}")
