# pyright: ignore
"""
AR.I.S — Universal Dependencies (UD v2.18) Streaming Ingestion Engine

Streams, tokenizes, and extracts structural linguistic samples from standard
Universal Dependencies CoNLL-U treebanks without requiring gigabytes of uncompressed disk space.

Constitutional Invariants:
  1. ZERO HARD PATCHES (Rule 1): Neutral linguistic extraction preserving original evidence.
  2. STREAMING I/O: Reads directly from compressed tar archives on-the-fly.
  3. UNIVERSAL SCHEMA: Standardized 10-column CoNLL-U representation.
"""

import os
import tarfile
from typing import Dict, Any, List, Optional, Iterator, Tuple
from dataclasses import dataclass, field


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
    def from_conllu_line(cls, line: str) -> Optional["UDToken"]:
        parts = line.strip().split("\t")
        if len(parts) < 10:
            return None

        # Skip multi-word contraction tokens (e.g. "1-2", "3-4") or empty nodes ("1.1")
        if "-" in parts[0] or "." in parts[0]:
            return None

        try:
            token_id = int(parts[0])
            head_id = int(parts[6]) if parts[6] != "_" and parts[6].isdigit() else 0
        except ValueError:
            return None

        # Parse morphological features (e.g. "Number=Sing|Person=3|Tense=Past")
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
    """Represents a full grammatically annotated sentence from Universal Dependencies."""
    sent_id: str
    text: str
    language: str
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


class ArisUDIngestionEngine:
    """
    High-performance streaming parser for Universal Dependencies v2.18 treebanks.
    """

    DEFAULT_TGZ_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Universal Dependencies 2.18",
        "ud-treebanks-v2.18.tgz"
    )

    # Supplemental CoNLL-U file: hand-annotated exemplars that extend the treebank
    # without modifying the core archive. Yielded first so the model absorbs them
    # before any treebank data, ensuring these patterns anchor the UPOS manifolds.
    SUPPLEMENTAL_CONLLU_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "ud_supplemental.conllu"
    )

    @classmethod
    def stream_treebank(
        cls,
        language_filter: Optional[str] = None,
        treebank_name: Optional[str] = None,
        split: str = "train",  # "train" | "dev" | "test" | "all"
        max_sentences: Optional[int] = None,
        tgz_path: Optional[str] = None,
    ) -> Iterator[UDSentence]:
        """
        Streams UDSentence objects from the supplemental CoNLL-U file first,
        then from the compressed UD tarball. Both sources produce identical
        UDSentence objects — the engine cannot distinguish origin.
        """
        yielded_count = 0

        # --- Supplemental CoNLL-U sentences (yielded before treebank data) ---
        if os.path.exists(cls.SUPPLEMENTAL_CONLLU_PATH):
            yield from cls._stream_conllu_file(
                cls.SUPPLEMENTAL_CONLLU_PATH,
                language="English",
            )

        archive_path = tgz_path or cls.DEFAULT_TGZ_PATH
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Universal Dependencies archive not found: {archive_path}")

        yielded_count = 0

        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".conllu"):
                    continue

                # Filter by treebank name or language
                if treebank_name and treebank_name.lower() not in member.name.lower():
                    continue
                if language_filter and language_filter.lower() not in member.name.lower():
                    continue
                if split != "all" and f"-ud-{split}.conllu" not in member.name:
                    continue

                # Extract language tag from folder (e.g. "UD_English-EWT" -> "English")
                folder_part = member.name.split("/")[1] if "/" in member.name else "UD_Unknown"
                lang_code = folder_part.replace("UD_", "").split("-")[0]
                filename = member.name.split("/")[-1]
                treebank_id = filename.split("-ud-")[0] if "-ud-" in filename else folder_part

                extracted_file = tf.extractfile(member)
                if not extracted_file:
                    continue

                curr_metadata: Dict[str, str] = {}
                curr_tokens: List[UDToken] = []

                for raw_line in extracted_file:
                    line = raw_line.decode("utf-8", errors="replace").strip()

                    if not line:
                        # Sentence boundary
                        if curr_tokens:
                            sent_id = curr_metadata.get("sent_id", f"sent_{yielded_count}")
                            text = curr_metadata.get("text", " ".join(t.form for t in curr_tokens))
                            curr_metadata.setdefault("treebank", treebank_id)
                            sentence = UDSentence(
                                sent_id=sent_id,
                                text=text,
                                language=lang_code,
                                tokens=curr_tokens,
                                metadata=curr_metadata,
                            )
                            yield sentence
                            yielded_count += 1
                            if max_sentences and yielded_count >= max_sentences:
                                return

                            curr_metadata = {}
                            curr_tokens = []
                        continue

                    if line.startswith("#"):
                        # Comment / Metadata line
                        if "=" in line:
                            parts = line[1:].strip().split("=", 1)
                            curr_metadata[parts[0].strip()] = parts[1].strip()
                        continue

                    tok = UDToken.from_conllu_line(line)
                    if tok:
                        curr_tokens.append(tok)

                # Trailing sentence
                if curr_tokens:
                    sent_id = curr_metadata.get("sent_id", f"sent_{yielded_count}")
                    text = curr_metadata.get("text", " ".join(t.form for t in curr_tokens))
                    curr_metadata.setdefault("treebank", treebank_id)
                    yield UDSentence(
                        sent_id=sent_id,
                        text=text,
                        language=lang_code,
                        tokens=curr_tokens,
                        metadata=curr_metadata,
                    )
                    yielded_count += 1
                    if max_sentences and yielded_count >= max_sentences:
                        return

    @classmethod
    def _stream_conllu_file(
        cls,
        file_path: str,
        language: str = "English",
    ) -> Iterator[UDSentence]:
        """
        Streams UDSentence objects from a plain CoNLL-U file.
        Produces the same UDSentence schema as stream_treebank so
        train_from_treebanks cannot distinguish supplemental from treebank data.
        """
        curr_metadata: Dict[str, str] = {}
        curr_tokens: List[UDToken] = []
        sent_count = 0

        with open(file_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()

                if not line:
                    if curr_tokens:
                        sent_id = curr_metadata.get("sent_id", f"supp_{sent_count}")
                        text = curr_metadata.get("text", " ".join(t.form for t in curr_tokens))
                        curr_metadata.setdefault("treebank", "ud_supplemental")
                        yield UDSentence(
                            sent_id=sent_id,
                            text=text,
                            language=language,
                            tokens=curr_tokens,
                            metadata=curr_metadata,
                        )
                        sent_count += 1
                        curr_metadata = {}
                        curr_tokens = []
                    continue

                if line.startswith("#"):
                    if "=" in line:
                        parts = line[1:].strip().split("=", 1)
                        curr_metadata[parts[0].strip()] = parts[1].strip()
                    continue

                tok = UDToken.from_conllu_line(line)
                if tok:
                    curr_tokens.append(tok)

        # Trailing sentence (file without trailing blank line)
        if curr_tokens:
            sent_id = curr_metadata.get("sent_id", f"supp_{sent_count}")
            text = curr_metadata.get("text", " ".join(t.form for t in curr_tokens))
            curr_metadata.setdefault("treebank", "ud_supplemental")
            yield UDSentence(
                sent_id=sent_id,
                text=text,
                language=language,
                tokens=curr_tokens,
                metadata=curr_metadata,
            )
