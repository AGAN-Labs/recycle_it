from django.urls import path

from . import views

app_name = 'images'

urlpatterns = [
    path('', views.ImageView.as_view(), name='images'),
]