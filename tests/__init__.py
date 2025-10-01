# Test suite initialization for EEG-MCI-Bench
import sys
import os

# Add project root to path to avoid naming conflicts with 'code' stdlib module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)