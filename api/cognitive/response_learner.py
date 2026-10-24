"""
AR.I.S - Human-Language Response Learner Primitive

Converts a structured reasoning result into a natural-language response
sentence that a human can read.

Strict Non-Negotiable Directives:
  - ZERO relation-specific if/else (Rule 4).
  - ZERO question memorization or question-string access.
  - ZERO reasoning -- takes a fully-resolved ReasoningResult; does NOT re-derive answers.
  - Selection logic driven entirely by answer_slot (object | subject | boolean | unknown)
    and answer presence -- never by relation name.
  - Relation phrases are generated generically from the canonical name:
        rel.replace('_', ' ')  ->  'works at', 'located in', 'brother of', etc.
  - Sentence templates are parameterized, not hardcoded per question.
"""

from typing import Optional


class ArisResponseLearner:
    """
    Natural-Language Generation (NLG) realization primitive.

    Accepts a structured reasoning result dict and verbalizes it into
    a human-readable response without performing any reasoning.

    Input schema (all fields optional; learner handles any missing gracefully):
        {
            'subject':        str | None   -- semantic subject entity
            'relation':       str | None   -- canonical relation name (e.g. 'works_at')
            'object':         str | None   -- semantic object entity
            'answer_slot':    str | None   -- 'object' | 'subject' | 'boolean'
            'final_answer':   str | None   -- resolved answer value (entity or 'Yes'/'No')
            'reasoning_path': list         -- [[s, r, o], ...] hop provenance
        }
    """

    @classmethod
    def generate(cls, reasoning_result: dict) -> str:
        """
        Returns a human-readable response string for the given reasoning result.
        Routes through the canonical ArisGenerativeDecoder and neural conditioned decoder.
        """
        try:
            from .generative_decoder import ArisGenerativeDecoder
            res = ArisGenerativeDecoder.decode(reasoning_result)
            text = res.get("text", "")
            if text:
                return text
        except Exception:
            pass

        subject      = cls._clean(reasoning_result.get('subject'))
        relation     = cls._clean(reasoning_result.get('relation') or
                                  reasoning_result.get('predicted_canonical_relation'))
        obj          = cls._clean(reasoning_result.get('object'))
        answer_slot  = cls._clean(reasoning_result.get('answer_slot'))
        final_answer = cls._clean(reasoning_result.get('final_answer'))
        path         = reasoning_result.get('reasoning_path') or []

        rel_phrase = cls._relation_phrase(relation)

        if not final_answer:
            return cls._unknown_response(subject, rel_phrase, obj, answer_slot)

        if answer_slot == 'boolean':
            return cls._boolean_response(subject, rel_phrase, obj, final_answer)

        if len(path) > 1:
            return cls._multihop_response(path, final_answer)

        if answer_slot == 'object':
            return cls._object_response(subject, rel_phrase, final_answer)

        if answer_slot == 'subject':
            return cls._subject_response(final_answer, rel_phrase, obj)

        return f'The answer is {final_answer}.'

    @staticmethod
    def _object_response(subject: str, rel_phrase: str, answer: str) -> str:
        parts = [p for p in [subject, rel_phrase, answer] if p]
        sentence = ' '.join(parts).strip().rstrip('.')
        return sentence + '.' if sentence else f'The answer is {answer}.'

    @staticmethod
    def _subject_response(answer: str, rel_phrase: str, obj: str) -> str:
        parts = [p for p in [answer, rel_phrase, obj] if p]
        sentence = ' '.join(parts).strip().rstrip('.')
        return sentence + '.' if sentence else f'The answer is {answer}.'

    @staticmethod
    def _boolean_response(subject: str, rel_phrase: str, obj: str, answer: str) -> str:
        affirmative = answer.strip().lower() == 'yes'
        parts = [p for p in [subject, rel_phrase, obj] if p]
        fact = ' '.join(parts).strip()
        if affirmative:
            return f'Yes, {fact}.' if fact else 'Yes.'
        else:
            return f'No, based on current knowledge, {fact} is not confirmed.' if fact else 'No.'

    @staticmethod
    def _multihop_response(path: list, final_answer: str) -> str:
        hop_phrases = []
        for hop in path:
            if not isinstance(hop, (list, tuple)) or len(hop) < 3:
                continue
            s, r, o = str(hop[0]), str(hop[1]), str(hop[2])
            rel_phrase = r.replace('_', ' ')
            hop_phrases.append(f'{s} {rel_phrase} {o}')
        if not hop_phrases:
            return f'The answer is {final_answer}.'
        chain = ', and '.join(hop_phrases)
        return f'{chain} -- therefore {final_answer}.'

    @staticmethod
    def _unknown_response(subject: str, rel_phrase: str, obj: str, answer_slot: str) -> str:
        if subject and rel_phrase and obj:
            return (f'Based on current knowledge, I could not confirm the relationship '
                    f'between {subject} and {obj}.')
        if subject and rel_phrase:
            return f'Based on current knowledge, I could not determine what {subject} {rel_phrase}.'
        return 'Based on current knowledge, I was unable to answer this question.'

    @staticmethod
    def _clean(value) -> str:
        if not value:
            return ''
        return str(value).strip()

    @staticmethod
    def _relation_phrase(relation: str) -> str:
        if not relation:
            return ''
        return relation.replace('_', ' ')
