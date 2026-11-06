import sys
from pathlib import Path

# the skill's scripts are standalone files next to its SKILL.md, not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills" / "unsloth-lora" / "scripts"))
