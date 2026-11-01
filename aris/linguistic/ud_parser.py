"""
AR.I.S — Universal Dependencies (UD v2) CoNLL-U Parser & Linguistic Substrate
Purpose:
  Parses standard 10-column Universal Dependencies CoNLL-U treebank formats,
  extracting grammatical tokens (FORM, LEMMA, UPOS, FEATS, HEAD, DEPREL)
  and full dependency trees without external dictionary dependencies.
"""

from __future__ import annotations
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple


@dataclass
class UDToken:
    """Represents a single word/token in a Universal Dependencies sentence."""
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: Dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = "root"
    deps: str = "_"
    misc: str = "_"

    @classmethod
    def from_conllu_line(cls, line: str) -> Optional[UDToken]:
        parts = line.strip().split("\t")
        if len(parts) < 10:
            return None

        # Skip multi-word contraction tokens (e.g. "1-2") or empty nodes ("1.1")
        if "-" in parts[0] or "." in parts[0]:
            return None

        try:
            token_id = int(parts[0])
            head_id = int(parts[6]) if parts[6] != "_" and parts[6].isdigit() else 0
        except ValueError:
            return None

        feats_dict = {}
        if parts[5] != "_":
            for feat in parts[5].split("|"):
                if "=" in feat:
                    k, v = feat.split("=", 1)
                    feats_dict[k.strip()] = v.strip()

        return cls(
            id=token_id,
            form=parts[1],
            lemma=parts[2],
            upos=parts[3],
            xpos=parts[4],
            feats=feats_dict,
            head=head_id,
            deprel=parts[7],
            deps=parts[8],
            misc=parts[9],
        )


@dataclass
class UDSentence:
    """Represents a complete grammatically annotated sentence from Universal Dependencies."""
    sent_id: str
    text: str
    tokens: List[UDToken] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def words(self) -> List[str]:
        return [t.form for t in self.tokens]

    @property
    def lemmas(self) -> List[str]:
        return [t.lemma for t in self.tokens]

    @property
    def upos_tags(self) -> List[str]:
        return [t.upos for t in self.tokens]

    @property
    def dependencies(self) -> List[Tuple[int, int, str]]:
        """Returns list of (head_id, dependent_id, relation_label)."""
        return [(t.head, t.id, t.deprel) for t in self.tokens]


class ArisUDParser:
    """
    Parser and streaming loader for Universal Dependencies datasets (.conllu and .tgz archives).
    """

    @classmethod
    def parse_conllu_text(cls, text_lines: Iterator[str]) -> Iterator[UDSentence]:
        """
        Parses an iterator of CoNLL-U lines into structured UDSentence objects.
        """
        current_tokens: List[UDToken] = []
        current_meta: Dict[str, str] = {}
        sent_id = ""
        sent_text = ""

        for line in text_lines:
            line = line.strip()
            if not line:
                if current_tokens:
                    yield UDSentence(
                        sent_id=sent_id or f"sent_{len(current_tokens)}",
                        text=sent_text or " ".join(t.form for t in current_tokens),
                        tokens=current_tokens,
                        metadata=current_meta,
                    )
                    current_tokens = []
                    current_meta = {}
                    sent_id = ""
                    sent_text = ""
                continue

            if line.startswith("#"):
                # Metadata line (e.g., # sent_id = 1, # text = Hello world.)
                comment = line[1:].strip()
                if "=" in comment:
                    k, v = comment.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    current_meta[k] = v
                    if k == "sent_id":
                        sent_id = v
                    elif k == "text":
                        sent_text = v
                continue

            tok = UDToken.from_conllu_line(line)
            if tok:
                current_tokens.append(tok)

        if current_tokens:
            yield UDSentence(
                sent_id=sent_id or f"sent_{len(current_tokens)}",
                text=sent_text or " ".join(t.form for t in current_tokens),
                tokens=current_tokens,
                metadata=current_meta,
            )

    @classmethod
    def parse_file(cls, filepath: Path) -> Iterator[UDSentence]:
        """Parses a local .conllu text file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CoNLL-U file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            yield from cls.parse_conllu_text(f)

    @classmethod
    def stream_from_archive(cls, archive_path: Path, language: Optional[str] = None) -> Iterator[UDSentence]:
        """
        Streams sentences directly from a compressed ud-treebanks-v2.xx.tgz archive.
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"UD archive not found: {archive_path}")

        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".conllu"):
                    continue
                if language and language.lower() not in member.name.lower():
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                lines = (line.decode("utf-8", errors="replace") for line in f)
                yield from cls.parse_conllu_text(lines)

