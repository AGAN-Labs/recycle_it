import unittest
import platform
import pathlib

from selenium import webdriver

class RecycleitFunctionalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.args = None
        self.browser = webdriver.Firefox()

        if 'darwin' in platform.platform().lower():
            print('Using Mac OS X settings!')

    def tearDown(self) -> None:
        self.browser.quit()

    def test_recycle_it(self):
        # Alex hears about the new platform called recycle_it
        # He goes online to check out their webpage
        self.browser.get('http://localhost:8000')

        # He notices the page title and header mention the company name
        self.assertIn('Recycle It', self.browser.title)
        self.fail('Finish the test!')

        # He sees the login button


if __name__ == '__main__':
    unittest.main(warnings='ignore')


if __name__ == '__main__':
    unittest.main(warnings='ignore')
