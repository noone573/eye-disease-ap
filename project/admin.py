from django.contrib import admin
from .models import ImagePrediction, PredictionImage

admin.site.register(ImagePrediction)

@admin.register(PredictionImage)
class ImagePredictionAdmin(admin.ModelAdmin):
    list_display = ('image', 'result', 'timestamp')
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)