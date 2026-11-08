from harness.tools import format_results


def test_format_results_includes_urls():
    out = format_results(
        [
            {"title": "ObjObsSAP - VO Labs", "url": "https://vo-labs.dev/objobssap", "content": "An   ObjObsSAP\nservice ..."},
            {"title": "WD", "url": "https://www.ivoa.net/documents/ObjObsSAP/", "content": "x" * 1000},
        ]
    )
    assert "https://vo-labs.dev/objobssap" in out and "1. ObjObsSAP - VO Labs" in out
    assert "An ObjObsSAP service" in out  # whitespace collapsed
    assert len(out.split("2. WD")[1]) < 500  # snippet truncated


def test_format_results_empty():
    assert format_results([]) == "No results."
