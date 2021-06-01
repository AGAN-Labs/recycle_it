import unittest
import platform
import recycle.main as main

class RecycleitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.args = None
        if 'darwin' in platform.platform().lower():
            print('Using Mac OS X settings!')

    def test_input_data_exists(self):
        #checks that the object we are looking for is not None
        input_data = main.get_input_data()
        self.assertNotEqual(input_data, None)

    def test_input_against_model(self):
        test_func = main.run_input_against_model
        input_data = None
        model = None
        results = test_func(input_data = input_data, model = model)
        self.assertNotIsInstance(results, Exception)



if __name__ == '__main__':
    unittest.main()
