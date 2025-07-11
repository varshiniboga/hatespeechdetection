#!/usr/bin/env python3
"""
Script to run all unit tests for the hate speech detection project.
"""

import sys
import os
import unittest

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_tests():
    """Run all unit tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), '..', 'test')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
