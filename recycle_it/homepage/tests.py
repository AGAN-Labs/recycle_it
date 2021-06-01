from django.test import TestCase
from django.urls import resolve
from homepage import views

# Create your tests here.
class HomePageTest(TestCase):
    """
    Testing class-based views
    https://docs.djangoproject.com/en/3.2/topics/testing/advanced/
    """

    def test_root_url_resolves_to_home_page_view(self):
        # resolve is the function Django uses internally to resolve URLs
        # and find what view function they should map to.
        # We’re checking that resolve, when called with “/”,
        # the root of the site, finds a function called home_page.
        found_view_func = resolve('/')
        self.assertEqual(found_view_func.view_name, 'homepage:homepage')