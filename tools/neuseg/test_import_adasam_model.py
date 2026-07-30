from importlib import import_module
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
m = import_module('adasam.model.adasam_model')
print('Imported AdaSAMModel:', hasattr(m, 'AdaSAMModel'))
print('Imported PrototypeFiLM:', hasattr(m, 'PrototypeFiLM'))
