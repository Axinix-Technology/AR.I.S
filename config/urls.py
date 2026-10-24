# pyright: ignore
"""
AR.I.S — Main URL Routing Configuration
"""

from django.contrib import admin
from django.urls import path, include
from api.views import VisualizerPageView, TrainingLabPageView
from api.views_chat import ChatPageView

urlpatterns = [
    path('admin/', admin.site.urls),
    # AR.I.S REST API Gateway
    path('api/', include('api.urls')),
    # Direct access to AR.I.S Real-Time Cognitive Visualizer, Training Lab & Chat Interface
    path('visualizer/', VisualizerPageView.as_view(), name='visualizer-root'),
    path('training-lab/', TrainingLabPageView.as_view(), name='training-lab-root'),
    path('chat/', ChatPageView.as_view(), name='chat-root'),
]
