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
        # Alice hears about the new platform called recycle_it
        # We go online to check out their webpage
        self.browser.get('http://localhost:8000/')

        # We notice the page title and header mention the company name
        self.assertIn('Recycle It', self.browser.title)
        #self.fail('Finish the test!')

        # We see the login button


if __name__ == '__main__':
    unittest.main(warnings='ignore')

