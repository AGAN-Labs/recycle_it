import unittest
import platform
import pathlib

class RecycleitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.args = None
        if 'darwin' in platform.platform().lower():
            print('Using Mac OS X settings!')

    def test_tocyen_online_has_title(self):
        self.fail("Failed")


if __name__ == '__main__':
    unittest.main()
