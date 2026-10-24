# Django REST API Gateway

## Overview
The Django REST API Gateway serves as the foundational communication layer for **AR.I.S (Artificial Intelligent System)**. It exposes structured, secure REST endpoints allowing human users and external interfaces to connect, verify system health, and interact with the AI brain.

## Architecture
- **Framework**: Django 6.x + Django REST Framework 3.x
- **Environment Management**: Python `venv` + `python-dotenv` for configuration separation.
- **Core Package**: `config/`
  - `config/settings.py`: Centralized configuration loaded dynamically from `.env`.
  - `config/urls.py`: Root URL routing mapping `/api/` to the `api` app.
- **Application**: `api/` (Django app name, unchanged in INSTALLED_APPS)
  - `api/views.py`: REST endpoints delegating to `api.cognitive.brain.ArisBrain`.
  - `api/urls.py`: Maps routes for gateway connectivity and fact/relational endpoints.
  - `api/db.py`: MongoDB connection utilities.
  - `api/cognitive/`: Reasoning, memory, and learning engines (brain, epistemic_state, cognitive_learner, relational_beliefs, response_learner).
  - `api/linguistic/`: Tokenizer, syntactic normalizer, frame dataset, neural frame classifier.
  - `api/relational/`: Relational binder, retrieval, symbolic graph, GNN reasoner, structural advisor, relation abstraction/evaluator.
  - `api/semantic/`: Semantic encoder, semantic evaluator, query representation/evaluator, executable tools.
  - `api/tests.py`: Automated audit test suite (53 tests, all passing post-restructure).

## Endpoints & Web Interfaces
| Path | Method | Purpose | Response |
|---|---|---|---|
| `/api/` | `GET` | API Gateway Root | Status, timestamp, MongoDB health, client IP |
| `/api/connect/` | `GET` | User Connection Handshake | Status, timestamp, MongoDB health, client IP |
| `/api/status/` | `GET` | System Health Check | Status, timestamp, MongoDB health, client IP |
| `/chat/` | `GET` | AR.I.S Conversational Chat UI | Modern Chat Interface (`aris_chat.html`) |
| `/visualizer/` | `GET` | Cognitive Trace Visualizer UI | 8-Stage Real-time animated pipeline UI |
| `/training-lab/` | `GET` | Cognitive Training Lab UI | 14-Tier benchmark and evaluation UI |
| `/api/chat/` | `POST` | Multi-turn Dialogue Dialogue API | Dialogue response, coreference, salience |
| `/api/chat/sessions/<id>/` | `GET`, `DELETE` | Dialogue Session Context Detail & Reset | JSON context representation |

## Design Philosophy (Rule 1 & Rule 4)
- **Zero Hard Patches**: The gateway handles database disconnection gracefully without crashing or returning unhandled 500 exceptions.
- **Generic Architecture**: Built as a clean, modular API gateway that can accommodate dynamic AI endpoints and cognitive pipelines without rigid procedural code.
