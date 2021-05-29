from django.shortcuts import render
from django.views import generic


# Create your views here.

# Notes on Class-based views
# https://docs.djangoproject.com/en/3.2/topics/class-based-views/intro/

class HomePageView(generic.TemplateView):
    template_name = 'homepage/home.html'