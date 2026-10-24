# pyright: ignore
"""
AR.I.S — Conversational Chat API & Interface Views (Phase 1)
"""

import uuid
from django.views import View
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .cognitive.brain import ArisBrain
from .cognitive.dialogue_context import ArisDialogueContextManager


class ChatDialogueView(APIView):
    """
    POST /api/chat/
    Stateful multi-turn conversational endpoint.
    Accepts: { session_id?: str, message: str, user_id?: str }
    """

    def post(self, request):
        message = request.data.get("message", "").strip()
        if not message:
            return Response(
                {"error": "Field 'message' is required and cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        session_id = request.data.get("session_id", "").strip() or str(uuid.uuid4())
        user_id = request.data.get("user_id", "default_user").strip()

        result = ArisBrain.process_dialogue_turn(
            session_id=session_id,
            utterance=message,
            speaker_role="USER",
            user_id=user_id,
        )

        return Response(result, status=status.HTTP_200_OK)


class ChatSessionDetailView(APIView):
    """
    GET /api/chat/sessions/<session_id>/
    DELETE /api/chat/sessions/<session_id>/
    """

    def get(self, request, session_id):
        context = ArisDialogueContextManager.get_or_create_context(session_id=session_id)
        return Response(context.to_dict(), status=status.HTTP_200_OK)

    def delete(self, request, session_id):
        cleared = ArisDialogueContextManager.clear_context(session_id=session_id)
        return Response({"session_id": session_id, "cleared": cleared}, status=status.HTTP_200_OK)


class ChatPageView(View):
    """
    GET /chat/
    Renders the modern, interactive AR.I.S Conversational Chat UI.
    """

    def get(self, request):
        return render(request, "aris_chat.html")
