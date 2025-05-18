from django.shortcuts import render, redirect, get_object_or_404
from .forms import PredictionForm
import os
import cv2
import tensorflow as tf
from django.conf import settings
from .models import PredictionImage
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from django.core.files import File
from django.views.decorators.http import require_POST

# Load model once globally
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'ml_models', 'vgg16_model.h5')
model = load_model(model_path)

class_names = ['Cataract', 'Retinopathy', 'Glaucoma', 'Normal']

def predict_image(img_path, threshold=0.60):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    predicted_index = np.argmax(preds)
    confidence = float(np.max(preds))

    if confidence < threshold:
        return "Not an eye", confidence, -1
    return class_names[predicted_index], confidence, predicted_index

def generate_gradcam(img_path, model, class_index, last_conv_layer_name='block5_conv3'):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Save to a unique path
    filename = f'gradcam_temp.jpg'
    gradcam_path = os.path.join(settings.MEDIA_ROOT, filename)
    cv2.imwrite(gradcam_path, superimposed_img)

    return gradcam_path

def home(request):
    return render(request, 'home.html')

def predict_view(request):
    prediction = None
    image_url = None
    gradcam_url = None
    predicted_class = None  
    confidence = None
    history = PredictionImage.objects.order_by('-timestamp')

    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()

            label, confidence, predicted_index = predict_image(obj.image.path)

            if confidence < 0.6:
                obj.result = "Not an eye"
                predicted_class = "Not an eye"
            else:
                obj.result = f"{label} ({confidence * 100:.2f}%)"
                predicted_class = label

                # Generate Grad-CAM and save
                gradcam_temp_path = generate_gradcam(obj.image.path, model, predicted_index)
                with open(gradcam_temp_path, 'rb') as f:
                    obj.gradcam.save(f'gradcam_{obj.pk}.jpg', File(f), save=False)
                gradcam_url = obj.gradcam.url

            obj.save()
            prediction = obj.result
            image_url = obj.image.url

    else:
        form = PredictionForm()

    return render(request, 'home.html', {
        'form': form,
        'prediction': prediction,
        'image_url': image_url,
        'gradcam_url': gradcam_url,
        'prediction_class': predicted_class,
        'confidence': f"{confidence * 100:.2f}%" if confidence is not None and predicted_class != "Not an eye" else None,
        'history': history,
    })

def view_history(request):
    history = PredictionImage.objects.all().order_by('-timestamp')
    return render(request, 'history.html', {'history': history})  # or edit_prediction.html

def delete_prediction(request, id):
    result = get_object_or_404(PredictionImage, id=id)
    result.delete()
    return redirect('view_history') 
