"""Append-only run registry with file locking."""
from datetime import datetime,timezone
from pathlib import Path
import json
from filelock import FileLock


def register(path,run_id,status,**fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.lock'):
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'run_id':run_id,'status':status,'at_utc':datetime.now(timezone.utc).isoformat(),**fields},ensure_ascii=False,sort_keys=True)+'\n')
