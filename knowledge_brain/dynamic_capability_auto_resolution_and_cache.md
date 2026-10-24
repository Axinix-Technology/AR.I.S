# Dynamic Capability Auto-Resolution & Self-Knowledge Cache

## 1. Overview & Architectural Motivation
In earlier iterations of AR.I.S, external tools were manually registered inside `ArisCapabilityRegistry._CAPABILITIES` as a hardcoded Python dictionary (`{"math_computation": ArisMathTool, ...}`). This created tight code-level coupling, required manual imports, and prevented the system from autonomously discovering and understanding its own tools and operational usage semantics.

Per the Non-Negotiable Directives (Rule 1: Zero Hard Patches; Rule 4: Generic Intelligence Over Code-Based Orchestration), `ArisCapabilityRegistry` now provides:
1. **Dynamic Subclass Auto-Resolution**: Automatically discovers all concrete subclasses of `ArisCapability` using Python runtime reflection, eliminating static dictionary declarations.
2. **Persistent Self-Knowledge Cache (`data/aris_capabilities_cache.json`)**: Persists structured capability specifications, schema contracts, and human-understandable usage descriptions.
3. **Intrinsic Self-Understanding (`get_capability_usage`)**: AR.I.S cognitive modules ([ArisBrain](file:///E:/Loigmax/AR.I.S/api/cognitive/brain.py), [ArisReasoningPlanner](file:///E:/Loigmax/AR.I.S/api/cognitive/deliberative_planner.py)) query the registry to know what capabilities exist, what they do, and when to invoke them without procedural hardcoding.

---

## 2. Capability Self-Knowledge Contract

Every capability inheriting from [ArisCapability](file:///E:/Loigmax/AR.I.S/api/semantic/executable_tools.py#L18) self-describes its operational semantics through standardized attributes:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `CAPABILITY_NAME` | `str` | Canonical identifier (e.g. `math_computation`, `browser_access`, `command_prompt`). |
| `ALIASES` | `List[str]` | Alternative invocation synonyms (e.g. `['terminal', 'shell', 'git_tool']`). |
| `IS_EXTERNAL` | `bool` | Flag declaring whether the capability incurs external network or system side-effects. |
| `AFFORDANCE_DOMAIN`| `str` | Semantic metric domain for continuous neural slot binding. |
| `DESCRIPTION` | `str` | Plain-language functional summary. |
| `USAGE` | `str` | Detailed operational guidance explaining when and how AR.I.S should activate the capability. |
| `PARAMETERS_SCHEMA`| `Dict[str, Any]`| Structured schema of required and optional parameters. |
| `ACCEPTS` | `Dict[str, Any]`| Invariant input contract (`domain`, `requires`). |
| `PRODUCES` | `Dict[str, Any]`| Invariant output contract (`state_transition`, `output_type`). |

---

## 3. Operational Mechanics

```
                  ┌───────────────────────────────┐
                  │    ArisCapability Subclasses   │
                  │ (Math, Browser, CommandPrompt) │
                  └───────────────┬───────────────┘
                                  │ Python Runtime Reflection
                                  ▼
                  ┌───────────────────────────────┐
                  │     ArisCapabilityRegistry    │
                  │  auto_resolve() / get_usage() │
                  └───────┬───────────────▲───────┘
     Synchronizes to JSON │               │ Reads on Startup
                          ▼               │
                  ┌───────────────────────────────┐
                  │ data/aris_capabilities_cache  │
                  │             .json             │
                  └───────────────────────────────┘
```

1. **Auto-Discovery**: `ArisCapabilityRegistry._auto_discover_subclasses()` traverses `ArisCapability.__subclasses__()` recursively.
2. **Cache Synchronization**: Automatically persists all discovered tools, schemas, and usage descriptions to `data/aris_capabilities_cache.json`.
3. **Dynamic Introspection**: New tools created dynamically at runtime (e.g. in test suites or extension plugins) are discovered automatically on query without restart or manual registration.
4. **Self-Knowledge API**:
   - `ArisCapabilityRegistry.get_capability_usage(name)`: Returns the plain-language usage instructions for AR.I.S reasoning.
   - `ArisCapabilityRegistry.get_all_capabilities_usage()`: Returns all active capability specifications.
   - `ArisCapabilityRegistry.get_capabilities_knowledge()`: Formats dynamic capability context for Teacher / Structural Advisor.

---

## 4. Verification & Audit Trail

Verified via automated test suites:
- [test_capability_auto_resolve.py](file:///E:/Loigmax/AR.I.S/api/tests/test_capability_auto_resolve.py): Confirms dynamic subclass discovery, cache JSON persistence, usage retrieval, and runtime capability registration (4 passed in 0.23s).
- [test_phase1_capability_affordance_perception.py](file:///E:/Loigmax/AR.I.S/api/tests/test_phase1_capability_affordance_perception.py): Confirms zero-execution affordance perception across auto-resolved registry (2 passed in 7.18s).
- [test_multi_tool_git_execution.py](file:///E:/Loigmax/AR.I.S/api/tests/test_multi_tool_git_execution.py): Confirms 3-step deliberative reasoning and execution on `E:\Loigmax\E-commerce` using auto-resolved capabilities (1 passed in 18.57s).
