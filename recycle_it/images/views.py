from django.shortcuts import render
from django.views import generic

# Create your views here.
class ImageView(generic.TemplateView):
    template_name = 'homepage/home.html'