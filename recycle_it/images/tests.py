from django.test import TestCase
from django.urls import resolve



# Create your tests here.
class ImagesTest(TestCase):
    """
    Testing class-based views
    https://docs.djangoproject.com/en/3.2/topics/testing/advanced/
    """

    def test_root_url_resolves_to_images_view(self):
        # resolve is the function Django uses internally to resolve URLs
        # and find what view function they should map to.
        # We’re checking that resolve, when called with “/”,
        # the root of the site, finds a function called home_page.
        found_view_func = resolve('/images')
        self.assertEqual(found_view_func.view_name, 'images:images')



