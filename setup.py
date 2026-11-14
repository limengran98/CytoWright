"""Install CytoWright and its model-development programs."""
from pathlib import Path
import shutil

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "cytowright" / "resources"
        for directory in ["configs", "assets", "programs"]:
            for source in (ROOT / directory).rglob("*"):
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                if source.relative_to(ROOT).as_posix() == "configs/local_endpoints.json":
                    continue
                if source.suffix not in ({".json"} if directory in {"configs", "assets"} else {".py", ".md"}):
                    continue
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


setup(
    name="cytowright",
    description="Agentic model development for cellular-response prediction",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["cytowright", "cytowright.*"]),
    install_requires=["h5py>=3.8", "numpy>=1.24", "pandas>=2.0", "rdkit>=2023.9", "torch>=2.1"],
    entry_points={"console_scripts": ["cytowright=cytowright.cli:main"]},
    cmdclass={"build_py": BuildPy},
)
