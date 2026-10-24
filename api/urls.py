# pyright: ignore
"""
AR.I.S — API URL Routing
"""

from django.urls import path
from .views import (
    ConnectionStatusView,
    BrainFactListCreateView,
    BrainFactDetailView,
    BrainRelationalQueryView,
    BrainTraceView,
    VisualizerPageView,
    TrainingLabPageView,
    TrainingLabRunView,
)
from .views_chat import (
    ChatDialogueView,
    ChatSessionDetailView,
    ChatPageView,
)

urlpatterns = [
    # Foundational API connection endpoints
    path('', ConnectionStatusView.as_view(), name='api-root'),
    path('connect/', ConnectionStatusView.as_view(), name='api-connect'),
    path('status/', ConnectionStatusView.as_view(), name='api-status'),

    # AR.I.S Brain Fact Storage & Retrieval Endpoints (Token-Ready)
    path('brain/facts/', BrainFactListCreateView.as_view(), name='brain-facts-list-create'),
    path('brain/facts/<str:fact_id>/', BrainFactDetailView.as_view(), name='brain-fact-detail'),

    # AR.I.S Brain Relational Retrieval (Tier 4)
    path('brain/query/', BrainRelationalQueryView.as_view(), name='brain-relational-query'),

    # AR.I.S Real-Time Cognitive Trace & Visualizer
    path('brain/trace/', BrainTraceView.as_view(), name='brain-trace'),
    path('visualizer/', VisualizerPageView.as_view(), name='brain-visualizer'),

    # AR.I.S Generalization Training Lab
    path('training-lab/', TrainingLabPageView.as_view(), name='brain-training-lab'),
    path('training-lab/run/', TrainingLabRunView.as_view(), name='brain-training-lab-run'),

    # AR.I.S Conversational Chat Agent
    path('chat/', ChatDialogueView.as_view(), name='api-chat-dialogue'),
    path('chat/sessions/<str:session_id>/', ChatSessionDetailView.as_view(), name='api-chat-session-detail'),
    path('chat/ui/', ChatPageView.as_view(), name='brain-chat-ui'),
]

