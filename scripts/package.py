"""Create the CytoWright source archive."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"cytowright", "configs", "assets", "programs", "examples", "docs", "scripts", "tests"}
ROOT_FILES = {"README.md", "README.zh-CN.md", "LICENSE", "pyproject.toml", "setup.py", "MANIFEST.in", ".gitignore"}


def selected_files():
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(part in {".git", "build", "dist", "__pycache__", ".venv"} or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo", ".log"} or relative.as_posix()=="configs/local_endpoints.json":
            continue
        if path.is_symlink():
            raise ValueError("Source package files must be ordinary files")
        if (len(relative.parts)==1 and path.name in ROOT_FILES) or relative.parts[0] in DIRECTORIES or relative.as_posix()=="data/README.md":
            result.append(path)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent/"CytoWright.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".building.zip")
    files = selected_files()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            member = zipfile.ZipInfo("CytoWright/"+path.relative_to(ROOT).as_posix(), date_time=(1980,1,1,0,0,0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, path.read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    temporary.replace(args.output)
    print("Created "+args.output.name+" with "+str(len(files))+" source files.")


if __name__=="__main__":
    main()
