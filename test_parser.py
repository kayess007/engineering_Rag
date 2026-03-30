import traceback

out = open("test_output.txt", "w", encoding="utf-8")

try:
    out.write("Importing parser...\n"); out.flush()
    from app.parser import parse_with_pymupdf
    out.write("Parser imported OK\n"); out.flush()

    elements = parse_with_pymupdf("storage/uploads/maintenance.pdf")
    out.write(f"Parsing complete. Total elements: {len(elements)}\n"); out.flush()

    titles = [e for e in elements if e["type"] == "Title"]
    tables = [e for e in elements if e["type"] == "Table"]
    narrative = [e for e in elements if e["type"] == "NarrativeText"]

    out.write(f"Titles: {len(titles)}\n")
    out.write(f"Tables: {len(tables)}\n")
    out.write(f"NarrativeText: {len(narrative)}\n\n")
    out.write("Sample titles:\n")
    for t in titles[:30]:
        out.write(f"  [{t['category_depth']}] p{t['page_number']}: {t['text'][:70]}\n")

    out.flush()

except Exception as e:
    out.write(f"ERROR: {e}\n")
    out.write(traceback.format_exc())
    out.flush()

out.close()
