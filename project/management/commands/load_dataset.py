from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.files import File
import os
from project.models import PredictionImage  # Replace with your model

class Command(BaseCommand):
    help = 'Load images from dataset folder into the database'

    def handle(self, *args, **kwargs):
        folder = os.path.join(settings.MEDIA_ROOT, 'dataset')
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                with open(os.path.join(folder, filename), 'rb') as f:
                    PredictionImage.objects.create(image=File(f), prediction_result='Pending')
        self.stdout.write(self.style.SUCCESS('Dataset loaded successfully!'))