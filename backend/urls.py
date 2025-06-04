from django.urls import path
from .views import predict_learning_path

urlpatterns = [
    path('predict-learning-path/', predict_learning_path, name='predict-learning-path'),
]
