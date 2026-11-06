import sys
from pathlib import Path

# the scripts are standalone files next to SKILL.md, not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "unsloth-ollama" / "scripts"))
