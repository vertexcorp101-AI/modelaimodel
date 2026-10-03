import os
import sys

# Hugging Face Spaces free runtime: app.py is the entrypoint and the
# platform sets $PORT (7860) + proxies it. The provider is dependency-free
# (stdlib only) so no build deps are needed.
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "7860")
os.environ.setdefault("LIGHTBRAIN_MODEL", "SmolLM2-360M-Instruct")

sys.argv = [sys.argv[0]]

from provider import main

main()