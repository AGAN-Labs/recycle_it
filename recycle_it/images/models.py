from django.db import models

# Create your models here.
class Images(models.Model):
    # user_images goes where?
    image_data = models.ImageField(upload_to='images_received', null=False)
    receive_time = models.DateTimeField(auto_now_add=True)
