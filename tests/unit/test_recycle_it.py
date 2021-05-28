import unittest
import platform
import pathlib

class RecycleitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.args = None
        if 'darwin' in platform.platform().lower():
            print('Using Mac OS X settings!')


if __name__ == '__main__':
    unittest.main()
