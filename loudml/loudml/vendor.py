"""
LoudML vendor environment
"""

import os
import sys

# RHEL
vendor_dir = os.path.join('/', 'usr', 'lib64', 'loudml', 'vendor')
sys.path.insert(0, vendor_dir)

# Debian
vendor_dir = os.path.join('/', 'usr', 'lib', 'loudml', 'vendor')
sys.path.insert(0, vendor_dir)
