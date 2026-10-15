#!/usr/bin/env python3
from pathlib import Path
import sys

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

from app.core import rebuild

dbp = base / 'data' / 'jarvis.db'
m = rebuild.verify_hashes(dbp)
if not m:
    print('ok')
else:
    for x in m:
        print(x)
