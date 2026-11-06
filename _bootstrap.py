"""Repository imports and relative paths."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
def local_path(value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def settings(path):
    import yaml
    return yaml.safe_load(local_path(path).read_text(encoding='utf-8'))
