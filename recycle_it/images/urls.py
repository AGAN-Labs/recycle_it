from django.urls import path, re_path

from . import views


app_name = 'images'

urlpatterns = [
    re_path(r'^upload/$', views.ImageView.as_view(), name='upload')
]

