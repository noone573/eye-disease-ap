from django.db import models

class EyeDiseasePrediction(models.Model):
    disease_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    description = models.TextField(blank=True, null=True)
    advice = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.disease_name} ({self.confidence * 100:.2f}%)"
    
class ImagePrediction(models.Model):
    prediction_result = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class PredictionImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    gradcam = models.ImageField(upload_to='gradcam_results/', null=True, blank=True)
    result = models.CharField(max_length=255, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.result} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"