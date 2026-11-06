"""Standalone script entrypoint, including when the package is not pip-installed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from qdpod.pipeline import main

if __name__=='__main__':
    main()
