"""``python -m helioseq`` -- same entry point as the ``helioseq`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
