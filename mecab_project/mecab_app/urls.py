from django.urls import path
from . import views

urlpatterns = [
    path('mecab_analyze/', views.mecab_analyze, name='mecab_analyze'),
]