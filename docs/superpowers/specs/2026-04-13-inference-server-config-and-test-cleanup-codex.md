# Inference Server Config Refactor And Test Cleanup

## Goal

Make the current inference-server branch PR-ready by fixing the scaling problem in the public serve API and by reducing test sprawl without reducing real coverage.

This design explicitly does **not** preserve backward compatibility. The API should be cleaned up around backend-owned typed config rather than carrying forward `backend="dynamo"`, `dynamo_config`, and other mixed generic/backend-specific fields.

## Current Problems

### 1. Public API bloat

Today the public serve API mixes generic and backend-specific concerns in the same objects:

- `InferenceServer` owns generic lifecycle fields plus backend-dispatch state and Dynamo-only infra knobs.
- `InferenceModelConfig` owns generic identity fields plus Ray Serve deployment fields, shared vLLM engine fields, and an untyped `dynamo_config` bag.

That shape does not scale to:

- additional Dynamo engines such as Dynamo-vLLM and Dynamo-SGLang
- future Dynamo features such as tool calling, ModelExpress, KVBM, and multimodal topology
- future non-Dynamo backends

Every new backend or feature would either:

- widen `InferenceServer`
- widen `InferenceModelConfig`
- add more undocumented keys to a backend-specific dict

### 2. Test sprawl

The branch introduces substantial new test coverage, but the layout is not review-friendly:

- source files do not map cleanly to test files
- many test classes are small organizational wrappers rather than fixtures for expensive setup
- validation cases are often split into multiple tiny tests that could be merged into scenario tests or parameterized tests
- some Ray-using tests bypass the shared session cluster
- some tests are marked `gpu` even when they only need Ray actors/process management logic

The objective is not fewer tests for its own sake. The objective is tighter tests that still exercise real logic and preserve or improve meaningful coverage.

## Design Principles

1. Backend-owned config, not generic config bags.
2. One `InferenceServer` instance owns exactly one backend family.
3. One Dynamo server owns exactly one Dynamo engine family.
4. One model config belongs to exactly one engine family.
5. Server-wide settings should exist once, not be repeated across models.
6. Test organization should mirror source organization.
7. Shared Ray test infrastructure should be reused everywhere unless a directory intentionally overrides it for a non-Ray test shape.

## Chosen Architecture

The chosen design is the typed-config variant corresponding to the earlier "Option C":

- `InferenceServer` accepts a typed backend config object, not a backend string and backend-specific kwargs.
- models use backend-specific model config classes rather than a generic model config with backend-specific bags.
- server-wide Dynamo router and infra settings live once on the Dynamo server config.
- `engine_kwargs` stays a raw dict, but it moves into backend- or engine-specific model config types instead of staying on the generic model config.
- one user-provided `runtime_env` lives once per server on the backend config, while backend implementations may still layer internal defaults or overrides at launch time.

This is the right shape even if Dynamo later supports both vLLM and SGLang, because "Dynamo" remains a backend family while the engine family is selected inside the Dynamo server config.

## Public API Shape

### Base model identity

```python
@dataclass
class BaseModelConfig:
    model_identifier: str
    model_name: str | None = None
```

These two fields stay generic because they are true model identity fields independent of backend.

### Ray Serve model config

```python
@dataclass
class RayServeModelConfig(BaseModelConfig):
    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
```

### Dynamo role config

```python
@dataclass
class DynamoRoleConfig:
    num_replicas: int = 1
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
```

### Dynamo vLLM model config

```python
@dataclass
class DynamoVLLMModelConfig(BaseModelConfig):
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    mode: Literal["agg", "disagg"] = "agg"
    num_replicas: int = 1
    prefill: DynamoRoleConfig | None = None
    decode: DynamoRoleConfig | None = None
    kv_events_config: dict[str, Any] = field(default_factory=dict)
```

This config deliberately holds only per-model settings. Router, namespace, discovery, and transport settings do not belong here.

### Ray Serve server config

```python
@dataclass
class RayServeServerConfig:
    family: ClassVar[Literal["ray_serve"]] = "ray_serve"
    runtime_env: dict[str, Any] = field(default_factory=dict)
```

### Dynamo router config

```python
@dataclass
class DynamoRouterConfig:
    mode: Literal["round-robin", "random", "kv", "direct"] | None = None
    kv_events: bool = True
    kv_overlap_score_weight: float = 1.0
    temperature: float = 0.0
    queue_threshold: int | None = None
    ttl_secs: float = 120.0
    max_tree_size: int = 2**20
    prune_target_ratio: float = 0.8
    reset_states: bool = False
```

### Dynamo server config

```python
@dataclass
class DynamoServerConfig:
    family: ClassVar[Literal["dynamo"]] = "dynamo"
    engine: Literal["vllm"] = "vllm"
    runtime_env: dict[str, Any] = field(default_factory=dict)
    etcd_endpoint: str | None = None
    nats_url: str | None = None
    namespace: str = DEFAULT_DYNAMO_NAMESPACE
    request_plane: str = DEFAULT_DYNAMO_REQUEST_PLANE
    event_plane: str = DEFAULT_DYNAMO_EVENT_PLANE
    router: DynamoRouterConfig = field(default_factory=DynamoRouterConfig)
```

When Dynamo-SGLang is added later, that should be represented by:

- extending `DynamoServerConfig.engine` to include `"sglang"`
- adding a `DynamoSGLangModelConfig`
- adding Dynamo-engine-specific implementation code inside the Dynamo backend family

It should **not** require new `InferenceServer` fields.

### InferenceServer

```python
@dataclass
class InferenceServer:
    backend: RayServeServerConfig | DynamoServerConfig
    models: list[RayServeModelConfig] | list[DynamoVLLMModelConfig]
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False
```

## Validation Rules

The new API should enforce the architecture directly rather than depending on convention.

### Backend/model family matching

- `InferenceServer(backend=RayServeServerConfig(...))` only accepts `RayServeModelConfig` instances.
- `InferenceServer(backend=DynamoServerConfig(engine="vllm", ...))` only accepts `DynamoVLLMModelConfig` instances.
- later `InferenceServer(backend=DynamoServerConfig(engine="sglang", ...))` should only accept `DynamoSGLangModelConfig` instances.

### Dynamo family rules

- one `InferenceServer` instance may only own one Dynamo engine family
- one model config may only own one engine family
- one Dynamo server owns one router configuration shared by every model in that server

### What disappears

The current cross-model frontend/router validation exists because the same settings are repeated in each model's `dynamo_config`. Once router and infra settings live once on `DynamoServerConfig`, that whole class of "models disagree about server-wide settings" validation disappears.

### What remains

These validations still matter:

- unique public model names
- sanitized Dynamo component-name collisions
- GPU inventory validation
- disaggregated serving role validation
- multi-node placement constraints
- backend/model family mismatch errors
- stable backend-family reporting for `_active_servers` and pipeline GPU-contention checks

## Placement Of Existing Fields

The specific fields discussed during design land here:

| Field | New owner |
| --- | --- |
| `model_identifier` | `BaseModelConfig` |
| `model_name` | `BaseModelConfig` |
| `engine_kwargs` | backend/engine-specific model config |
| `runtime_env` | backend/server config only |
| `deployment_config` | `RayServeModelConfig` |
| `dynamo_config` | removed and replaced by typed Dynamo config classes |
| `backend: str` | removed and replaced by typed backend config on `InferenceServer` |
| `etcd_endpoint` | `DynamoServerConfig` |
| `nats_url` | `DynamoServerConfig` |

## Package Layout

The current `internal/` directory is not buying much. The code is effectively imported and tested as a real package surface already, and the next round of Dynamo growth will benefit from clearer backend-family boundaries.

Proposed layout:

```text
nemo_curator/core/serve/
  __init__.py
  server.py
  config.py
  backends/
    shared/
      subprocess_mgr.py
      errors.py
      runtime_env.py
      types.py
    ray_serve/
      backend.py
    dynamo/
      backend.py
      vllm.py
```

### Boundary ownership

- `server.py`
  - owns high-level lifecycle
  - owns endpoint waiting and active-server tracking
  - dispatches to the selected backend implementation

- `config.py`
  - owns public typed config classes
  - owns backend/model compatibility validation entrypoints
  - is the only place public serve config dataclasses are defined

- `backends/ray_serve/*`
  - owns Ray Serve-specific translation from typed config into `LLMConfig` and Serve deployment behavior

- `backends/dynamo/*`
  - owns Dynamo-specific orchestration
  - owns translation from typed Dynamo config into subprocess launches
  - `vllm.py` isolates Dynamo-vLLM launch/build logic from the rest of the backend

- `backends/shared/*`
  - owns only the genuinely reusable actor/process/placement/runtime-env helpers

## Responsibility Changes In Existing Code

### `server.py`

`server.py` should stop understanding Dynamo-specific concepts beyond dispatch and generic lifecycle. It should not carry:

- backend strings
- Dynamo infra fields
- backend-specific config parsing

It still needs a stable backend-family identifier for active-server tracking and for pipeline contention checks. That identifier should come from the typed backend config, for example `server.backend.family`, rather than from a user-supplied backend string.

### Ray Serve translation

`InferenceModelConfig.to_llm_config()` is Ray Serve-specific behavior today. That logic should move into the Ray Serve backend area rather than stay on a generic model config type.

Ray Serve should continue to merge backend-owned runtime environment with its internal quiet/logging overrides at translation time.

### Dynamo translation

Current Dynamo code reads repeated server-wide settings from model-level dicts. After the refactor:

- server-wide Dynamo settings come from `DynamoServerConfig`
- per-model settings come from `DynamoVLLMModelConfig`
- role-level settings come from `DynamoRoleConfig`

That means helpers like current router-resolution code become simpler and narrower.

Dynamo should continue to merge backend-owned runtime environment with backend-added defaults such as `ai-dynamo[vllm]` injection inside the backend implementation rather than exposing those internal defaults in the public config surface.

## PR Decomposition

The branch should be split into stacked PRs in this order:

### PR 1: Serve API and package boundary refactor

Scope:

- introduce typed backend/server/model configs
- move package structure toward `backends/{shared,ray_serve,dynamo}`
- remove `dynamo_config` and backend string dispatch from the public API
- keep behavior equivalent where possible without preserving old signatures

Why first:

- this is the branch-wide scaling fix
- every later backend feature is easier to review once the public shapes are sane

### PR 2: Shared subprocess/runtime infrastructure

Scope:

- stabilize and relocate shared subprocess/placement/runtime-env helpers
- keep them reusable by future backend families
- move subprocess isolation tests next to that shared code

Why separate:

- actor/process lifecycle is its own concern
- this logic is valuable independently of Dynamo routing or topology work

### PR 3: Dynamo core serving on the new API

Scope:

- aggregated single-node and baseline lifecycle under the typed Dynamo API
- infra bring-up
- health checks
- runtime-env injection behavior

Why separate:

- gives reviewers a compact, working Dynamo baseline before topology complexity lands

### PR 4: Dynamo topology work

Scope:

- multi-model placement
- multi-node TP
- disaggregated serving
- placement and validation hardening

Why separate:

- this is the large scheduling/topology feature cluster
- it is easier to review once API and process infrastructure are already merged

### PR 5: Dynamo routing work

Scope:

- KV-aware routing
- router launch arguments
- any remaining frontend-routing-specific validation

Why separate:

- routing behavior is a distinct surface from infra and topology

## Test Strategy

### Test organization rules

1. One source file should have one primary unit-test file.
2. Integration tests should live separately from unit tests.
3. Test classes should mainly exist for expensive shared fixtures, not for one- or two-test grouping.
4. Validation permutations should be parameterized or folded into scenario tests when that improves clarity.
5. Coverage should be preserved by keeping logic-heavy tests and deleting only low-value shape/assertion tests.

### Target mapping

| Source | Primary test file |
| --- | --- |
| `config.py` | `tests/core/serve/test_config.py` |
| `server.py` | `tests/core/serve/test_server.py` |
| `backends/shared/subprocess_mgr.py` | `tests/core/serve/backends/shared/test_subprocess_mgr.py` |
| `backends/ray_serve/backend.py` | `tests/core/serve/backends/ray_serve/test_backend.py` |
| `backends/dynamo/backend.py` | `tests/core/serve/backends/dynamo/test_backend.py` |
| `backends/dynamo/vllm.py` | `tests/core/serve/backends/dynamo/test_vllm.py` |

### Integration mapping

| Behavior | Integration test file |
| --- | --- |
| real Ray Serve lifecycle | `tests/core/serve/integration/test_ray_serve.py` |
| real Dynamo lifecycle | `tests/core/serve/integration/test_dynamo.py` |

### Ray fixture policy

Shared Ray infrastructure should be reused everywhere Ray is involved:

- use `shared_ray_cluster` and `shared_ray_client`
- do not call `ray.init()` and `ray.shutdown()` inside general-purpose test helpers
- only override the root fixture in subtrees that intentionally do not use Ray at all

One concrete cleanup target is the helper in `tests/backends/utils.py`, which currently owns its own Ray initialization to create a detached actor. That helper should be refactored to rely on the shared session cluster rather than managing Ray lifecycle ad hoc.

### GPU marker policy

Only tests that truly need CUDA or a GPU-backed Ray cluster should be marked `gpu`.

Candidates to reevaluate:

- subprocess/actor lifecycle tests that only need Ray and OS subprocess behavior
- Dynamo liveness tests that use lightweight subprocesses rather than real model execution

Real model-serving integration tests should remain `gpu`.

### Test consolidation policy

Good reductions:

- merge multiple "this invalid input raises" tests into one scenario test with labeled sections
- parameterize repeated shape-preserving cases
- remove tests that only assert trivial dataclass storage or one-line formatting with no branching

Bad reductions:

- deleting placement, liveness, routing, or health-check tests that exercise real logic
- collapsing integration coverage into mocked unit tests

### Specific cleanup targets

- `tests/core/test_serve.py`
  - split into server unit tests, Ray Serve integration tests, and any backend-specific runtime-env tests that belong elsewhere

- `tests/core/test_runtime_env_subprocess_isolation.py`
  - move next to shared subprocess/runtime-env code

- `tests/core/serve/internal/test_dynamo.py`
  - split by concern and reduce the number of tiny organizational classes

- `tests/backends/test_integration.py`
  - keep backend parameterization
  - reduce duplication in broad output-shape assertions where a smaller number of scenario tests can cover the same behavior
  - remove helper-owned Ray lifecycle

- `tests/stages/deduplication/semantic/test_kmeans.py`
  - keep the expensive integration fixture pattern because it avoids rerunning a heavy workflow
  - collapse smaller permutation-only tests where appropriate

## Non-Goals

- preserving the old public API
- introducing typed `engine_kwargs` dataclasses in this refactor
- allowing mixed Dynamo engine families inside one server
- renaming `nemo_curator.core.serve` to a new top-level package in the same change

The `nemo_curator.core.serve` to `nemo_curator.serve` move can be considered later, but it is not required to solve the current bloat and PR-splitting problems.

## Risks And Mitigations

### Risk: API refactor and package move in one PR becomes too large

Mitigation:

- keep PR 1 focused on type boundaries and package ownership only
- defer behavior-heavy Dynamo topology/routing work to later stacked PRs

### Risk: typed configs accidentally recreate dict-bag bloat

Mitigation:

- keep server-wide settings on server config only
- keep per-model settings on model config only
- keep engine-specific concerns inside backend-family model types

### Risk: test cleanup loses meaningful coverage

Mitigation:

- remove only low-value duplication
- keep logic-heavy validation, placement, liveness, and integration tests
- verify coverage before and after the cleanup-oriented PRs

## Acceptance Criteria

This design is satisfied when:

1. `InferenceServer` no longer grows backend-specific kwargs.
2. generic model config no longer carries backend-specific dict bags.
3. a new backend family can be added without changing the `InferenceServer` signature.
4. a new Dynamo engine family can be added without widening the generic public API.
5. server-wide Dynamo router settings are defined once.
6. unit tests map cleanly to source files.
7. Ray-using tests consistently reuse shared Ray fixtures unless intentionally exempt.
8. test cleanup preserves meaningful logic coverage.
