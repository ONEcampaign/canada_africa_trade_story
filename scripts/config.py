from pathlib import Path


class Paths:
    """Class to store the paths to the data and output folders."""

    project = Path(__file__).resolve().parent.parent

    raw_data = project / "raw_data"
    hs_categories = raw_data / "hs_categories.json"
    pydeflate = raw_data / "pydeflate"

    output = project / "output"

    scripts = project / "scripts"
