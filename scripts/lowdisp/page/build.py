import json, pathlib
here = pathlib.Path(__file__).parent
data = json.load(open("/mnt/project-files/lowdisp-solver/data/data.json"))
html = (here / "page.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
out = pathlib.Path("/mnt/project-files/lowdisp-solver/field-solver-dispersion.html")
out.write_text(html); print(out, len(html))
