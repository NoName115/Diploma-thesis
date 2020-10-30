import os

TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_test_path(*path):
    return os.path.join(TESTS_ROOT, *path)
