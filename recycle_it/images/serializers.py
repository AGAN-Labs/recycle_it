from django.contrib.auth.models import User, Group
from rest_framework import serializers

from . import models

class ImagesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.Images
        fields = ('image_data', 'receive_time')
