from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name=''),
    path('process/', views.process_view, name='process'),
]