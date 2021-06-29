import unittest
import platform
import requests
import pathlib

from selenium import webdriver
import recycle_it.config as recycle_it_config

class RecycleitFunctionalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.args = None
        self.browser = webdriver.Firefox()
        self.module_dir = pathlib.Path(__file__).parent.parent.parent
        self.test_upload_file = recycle_it_config.test_image_path
        print(self.test_upload_file)
        if 'darwin' in platform.platform().lower():
            print('Using Mac OS X settings!')

    def tearDown(self) -> None:
        self.browser.quit()

    def test_recycle_it(self):
        # Alice hears about the new platform called recycle_it
        # We go online to check out their webpage
        url = 'http://localhost:8000/'
        self.browser.get(url)

        # We notice the page title and header mention the company name
        self.assertIn('Recycle It', self.browser.title)

        # We want to upload an image of an object to see if it is recyclable
        # We see a prompt asking us to take or upload an image

        # We know the endpoint for sending images
        image_endpoint = recycle_it_config.image_endpoint
        self.assertIn("/images", image_endpoint)

        # We attempt to POST the image to the image endpoint
        upload_url = url + 'images/upload/'
        files = {'image_data': open(self.test_upload_file, 'rb')}
        result = requests.post(upload_url, files=files)
        self.assertEqual(result.status_code, 201)


        # We see a button for uploading a selected image


if __name__ == '__main__':
    unittest.main(warnings='ignore')

