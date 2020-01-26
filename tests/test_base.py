import os
import sys
import unittest
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestBase(unittest.TestCase):
    def setUp(self):
        # RHEL
        vendor_dir = os.path.join('/', 'usr', 'lib64', 'loudml', 'vendor')
        sys.path.insert(0, vendor_dir)

        # Debian
        vendor_dir = os.path.join('/', 'usr', 'lib', 'loudml', 'vendor')
        sys.path.insert(0, vendor_dir)

    def test_numpy(self):
        import numpy as np
        self.assertEqual(np.version.version, "1.16.4")

    def test_scipy(self):
        import scipy
        self.assertEqual(scipy.__version__, "1.3.3")

    def test_tf(self):
        import tensorflow as tf
        self.assertEqual(tf.__version__, "1.13.2")

        # Simple hello world using TensorFlow

        # Create a Constant op
        # The op is added as a node to the default graph.
        #
        # The value returned by the constructor represents the output
        # of the Constant op.
        hello = tf.constant('Hello, TensorFlow!')

        # Start tf session
        sess = tf.Session()

        # Run the op
        self.assertEqual(sess.run(hello), b'Hello, TensorFlow!')
