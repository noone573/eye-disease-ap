from django.urls import path # type: ignore
from . import views
urlpatterns = [ 
    path('', views.predict_view, name='prediction' ),
    path('history/', views.view_history, name='view_history'),
    path('delete/<int:id>/', views.delete_prediction, name='delete_prediction'),
    
                ]