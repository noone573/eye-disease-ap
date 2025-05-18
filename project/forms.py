from django import forms
from .models import PredictionImage
class UploadImageForm(forms.Form):
    image = forms.ImageField()

class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredictionImage
        fields = ['image', 'result', 'gradcam']