"""Shared pytest fixtures and path setup."""

import os
import sys

# Add the project src directory to the import path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
