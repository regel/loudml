"""
LoudML vendor environment
"""

import os
import sys
vendor_dir = os.path.join('/', 'usr', 'lib64', 'loudml', 'vendor')
sys.path.insert(0, vendor_dir)
