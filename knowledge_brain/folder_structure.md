# Folder Structure

## Overview
The AR.I.S project follows a clean, cognitively-organized directory layout that separates the Django infrastructure layer from the cognitive intelligence sub-packages. The flat 30-file `api/` monolith was restructured into four bounded sub-packages by domain.

## Directory Layout

```
AR.I.S/
├── manage.py                        # Django entry point
├── .env / .env-example / .gitignore
├── requirements.txt
├── AGENTS.md / GEMINI.md / CLAUDE.md
├── db.sqlite3
│
├── config/                          # Django project configuration
│   ├── settings.py                  # Loaded from .env via python-dotenv
│   ├── urls.py                      # Root: /api/ → api.urls
│   ├── asgi.py / wsgi.py
│   └── __init__.py
│
├── api/                             # Django app (INSTALLED_APPS: 'api')
│   ├── __init__.py
│   ├── admin.py / apps.py / models.py
│   ├── serializers.py               # DRF input/output serializers
│   ├── views.py                     # REST endpoints (delegates to cognitive.brain)
│   ├── urls.py                      # /api/brain/facts/, /api/brain/query/, etc.
│   ├── db.py                        # MongoDB connection (get_mongo_db, check_mongo_connection)
│   ├── migrations/ / management/
│   │
│   ├── cognitive/                   # Reasoning, memory, learning core
│   │   ├── brain.py                 # ArisBrain — central cognitive memory interface
│   │   ├── cognitive_learner.py     # Tier 5 operation learning & autonomous reasoner
│   │   ├── epistemic_state.py       # KNOWN / DERIVABLE / ACQUIRABLE / UNSUPPORTED states
│   │   ├── relational_beliefs.py    # Human-in-the-loop belief revision & inhibition
│   │   └── response_learner.py      # Response pattern learning
│   │
│   ├── linguistic/                  # Tokenization, normalization, structural frames
│   │   ├── tokenizer.py             # ArisTokenizer — Tier 1 & Tier 2 deterministic tokenizer
│   │   ├── syntactic_normalizer.py  # Canonical argument-role normalization (mood, passive, genitives)
│   │   ├── frame_dataset.py         # Structural curriculum generator & frame vectorizer
│   │   └── neural_frame_classifier.py  # Local PyTorch frame parameter classifier (< 1ms CPU)
│   │
│   ├── relational/                  # Graph traversal, binding, GNN reasoning
│   │   ├── relational.py            # ArisRelationalBinder — Tier 3 subject/relation/object discovery
│   │   ├── relation_abstraction.py  # Prototype-centroid relational space & subword embeddings
│   │   ├── relation_evaluator.py    # Semantic relation abstraction benchmark evaluator
│   │   ├── retrieval.py             # ArisRetriever — Tier 4 deterministic 1-hop traversal
│   │   ├── symbolic_graph.py        # ArisSymbolicGraph — entity-relation graph store
│   │   ├── gnn_reasoner.py          # ArisGNNReasoner — multi-hop positional path scoring
│   │   └── structural_advisor.py    # ArisStructuralAdvisor — Layer 0 sensory receptor
│   │
│   └── semantic/                    # Semantic representation, query binding, evaluation
│       ├── semantic_representation.py   # ArisSemanticEncoder — continuous coordinate space
│       ├── semantic_evaluator.py        # Full pipeline evaluation protocol
│       ├── query_representation.py      # Natural-language → canonical semantic query
│       ├── query_evaluator.py           # Query binding benchmark evaluator
│       └── executable_tools.py          # ArisCapabilityRegistry & ArisMathTool
│
├── models/                          # Trained PyTorch model artifacts
│   └── aris_frame_classifier.pt     # Offline neural frame classifier weights
│
├── data/                            # Training, evaluation, and benchmark datasets
│   ├── training Dataset.json        # Knowledge triples (facts) training set
│   ├── training question dataset.json  # Supervised QA training set
│   ├── aris_training_facts.json
│   ├── aris_question_training.json
│   ├── aris_evaluation_results.json
│   ├── aris_query_binding_dataset.json
│   ├── aris_query_binding_eval.json
│   ├── aris_relation_abstraction_benchmark.json
│   └── aris_relation_abstraction_eval.json
│
└── knowledge_brain/                 # Living architectural knowledge base (flat, INDEX-navigated)
    ├── INDEX.md                     # Master A–Z index
    └── *.md                         # Individual concept & module records
```

## Cross-Package Import Convention

All modules use Python relative imports. Cross-package imports traverse using `..`:

| From package | Imports into | Example |
|---|---|---|
| `cognitive/` | `relational/` | `from ..relational.relational import ArisRelationalBinder` |
| `cognitive/` | `linguistic/` | `from ..linguistic.tokenizer import ArisTokenizer` |
| `relational/` | `linguistic/` | `from ..linguistic.tokenizer import ArisTokenizer` |
| `relational/` | `cognitive/` | `from ..cognitive.brain import ArisBrain` |
| `semantic/` | `relational/` | `from ..relational.symbolic_graph import ArisSymbolicGraph` |
| `semantic/` | `cognitive/` | `from ..cognitive.brain import ArisBrain` |
| Any sub-pkg | `api/` root | `from ..db import get_mongo_db` |

## Design Decisions

- **`api/` app name preserved**: Django `INSTALLED_APPS` still references `'api'`. Only internal module organization changed.
- **`db.py` stays at api/ root**: It is infrastructure, not a cognitive module.
- **`relational_beliefs.py` in `cognitive/`**: It handles human belief revision and inhibition — a cognitive operation, not a raw graph traversal.
- **`data/` at project root**: All dataset files are runtime-loaded by path, not imported as Python modules. Their default paths in CLI commands are prefixed with `data/`.
- **`knowledge_brain/` stays flat**: The `INDEX.md` is the navigation layer; sub-folders would add friction without benefit.

## Established Patterns

- Test mocks use the full sub-package path: `@patch('api.cognitive.brain.ArisBrain._get_collection')`
- Management command lazy imports use full sub-package path: `from api.semantic.semantic_evaluator import ...`
- The `neural_frame_classifier.py` model path uses 3 `dirname` levels: `dirname(dirname(dirname(abspath(__file__))))` to reach the project root from `api/linguistic/`.
