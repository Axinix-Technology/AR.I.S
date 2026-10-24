# pyright: ignore
"""
AR.I.S — API Serializers

Validation and presentation contracts for the API gateway boundary.
Encapsulates data transformation before passing to ArisBrain.
"""

from rest_framework import serializers


class FactInputSerializer(serializers.Serializer):
    """
    Validates user/client input when storing a new fact.
    """
    statement = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        error_messages={
            "required": "A factual statement is required.",
            "blank": "Fact statement cannot be blank."
        }
    )
    topic = serializers.CharField(
        required=False,
        default="general",
        allow_blank=True,
        trim_whitespace=True
    )
    tags = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        default=list
    )
    confidence = serializers.FloatField(
        required=False,
        default=1.0,
        min_value=0.0,
        max_value=1.0
    )
    metadata = serializers.DictField(
        required=False,
        default=dict
    )


class TokenPositionDetailsSerializer(serializers.Serializer):
    """Detailed multi-scale positional coordinates."""
    sequence_index = serializers.IntegerField()
    sentence_index = serializers.IntegerField()
    token_in_sentence = serializers.IntegerField()
    char_start = serializers.IntegerField()
    char_end = serializers.IntegerField()
    relative_pos = serializers.FloatField()


class TokenPositionalSerializer(serializers.Serializer):
    """Represents a discrete cognitive token with exact positional attributes (Tier 1)."""
    sequence_index = serializers.IntegerField()
    token = serializers.CharField()
    raw = serializers.CharField()
    type = serializers.CharField()
    position = TokenPositionDetailsSerializer()


class ConceptCandidateSerializer(serializers.Serializer):
    """Represents a concept or entity candidate span (Tier 2)."""
    index = serializers.IntegerField()
    text = serializers.CharField()
    normalized = serializers.CharField()
    candidate_type = serializers.CharField()
    token_indices = serializers.ListField(child=serializers.IntegerField())
    char_start = serializers.IntegerField()
    char_end = serializers.IntegerField()
    sentence_index = serializers.IntegerField()


class FactOutputSerializer(serializers.Serializer):
    """Represents a stored cognitive fact returned from ArisBrain."""
    id = serializers.CharField()
    statement = serializers.CharField()
    topic = serializers.CharField()
    tags = serializers.ListField(child=serializers.CharField())
    source = serializers.CharField()
    confidence = serializers.FloatField()
    metadata = serializers.DictField()
    token_count = serializers.IntegerField(required=False, default=0)
    tokens = TokenPositionalSerializer(many=True, required=False, default=list)
    candidate_count = serializers.IntegerField(required=False, default=0)
    candidates = ConceptCandidateSerializer(many=True, required=False, default=list)
    relation_count = serializers.IntegerField(required=False, default=0)
    relations = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()
