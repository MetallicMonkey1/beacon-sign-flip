"""
beacon_tracer.py

The beacon tracer component of the Beacon-Routed Localist Memory (BRLM) architecture.

Core idea:
    During a forward pass, inject a low-magnitude, unique beacon vector alongside
    the normal input. As it propagates, record which containers activate and in what
    order. Over many passes this builds a live, weighted path graph — a map of how
    knowledge flows through the network.

    This map is what makes incremental scaling and surgical editing possible:
    - New knowledge? Route it to low-usage containers.
    - Edit existing knowledge? Find exactly which containers own it.
    - Scale up? Add empty containers; the system populates them naturally.

Components:
    Container       — a single localist knowledge slot (nn.Module)
    BeaconTracer    — hook-based tracer that builds the live path map
    BeaconTrace     — dataclass representing one recorded path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import uuid


# ---------------------------------------------------------------------------
# Container — a single localist knowledge slot
# ---------------------------------------------------------------------------

class Container(nn.Module):
    """
    A discrete, addressable knowledge slot.

    Internally just a small MLP, but treated as an atomic unit by the tracer.
    Metadata (semantic_tag, usage_count) lives here, not scattered across
    a weight matrix.

    Args:
        dim:         Vector dimension (should match the rest of the network).
        hidden_dim:  Internal MLP width. Defaults to 4x dim (standard transformer ratio).
        semantic_tag: Optional human-readable label. Can be set post-hoc once
                      the tracer tells you what concept lives here.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, semantic_tag: str = ""):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.semantic_tag = semantic_tag
        self.usage_count = 0  # incremented by tracer

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: container modulates the signal, doesn't replace it
        return self.norm(x + self.net(x))

    def __repr__(self):
        tag = f'"{self.semantic_tag}"' if self.semantic_tag else "untagged"
        return f"Container(dim={self.dim}, tag={tag}, usage={self.usage_count})"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeaconTrace:
    """
    A single recorded forward-pass path.

    path: ordered list of (container_id, activation_strength) for every
          container that fired above threshold during this pass.
    """
    beacon_id: str
    step: int
    path: list[tuple[str, float]] = field(default_factory=list)

    def container_ids(self) -> list[str]:
        return [c for c, _ in self.path]

    def __repr__(self):
        route = " → ".join(f"{c}({s:.3f})" for c, s in self.path)
        return f"BeaconTrace(step={self.step}, path=[{route}])"


# ---------------------------------------------------------------------------
# BeaconTracer
# ---------------------------------------------------------------------------

class BeaconTracer:
    """
    Injects beacon signals during training and builds a live path map.

    Typical usage
    -------------
        tracer = BeaconTracer(dim=256)

        # Register every container once at construction time
        for cid, module in containers.items():
            tracer.register_container(cid, module)

        # Inside your training loop:
        for step, (x, y) in enumerate(dataloader):
            with tracer.trace(step=step) as beacon:
                # Add beacon to input — low magnitude, won't meaningfully
                # shift the loss, but leaves a traceable signal
                output = model(x + beacon.unsqueeze(0))

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # The path map updates automatically after each trace context exits.
            # Inspect it whenever:
            print(tracer.last_trace)
            print(tracer.path_graph)

    Design notes
    ------------
    - Hooks are non-invasive: no changes needed to Container.forward().
    - Beacon magnitude is kept small (default 1e-2) so gradient signal is
      not meaningfully perturbed. The beacon is a whisper, not a shout.
    - The path graph accumulates edge weights over the entire training run.
      Edge weight = mean activation strength of the two endpoints. Stronger
      edge = more reliably co-activated = knowledge genuinely flows there.
    - Thread safety: not guaranteed for multi-GPU / async dataloaders.
      Wrap trace() in a lock if needed.
    """

    def __init__(
        self,
        dim: int,
        magnitude: float = 1e-2,
        activation_threshold: float = 0.05,
    ):
        """
        Args:
            dim:                  Dimension of the beacon vector. Must match
                                  the input dimension of your containers.
            magnitude:            Scale of the injected beacon. Keep small.
            activation_threshold: Containers with mean output activation below
                                  this are considered "silent" and not logged.
        """
        self.dim = dim
        self.magnitude = magnitude
        self.threshold = activation_threshold

        # Live path map: src_container_id → dst_container_id → cumulative weight
        # This is the map. Everything else is derived from it.
        self.path_graph: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Full ordered history of every trace
        self.traces: list[BeaconTrace] = []

        # Registered containers (ordered — hook fires in registration order)
        self._containers: dict[str, Container] = {}
        self._hooks: list = []

        # Live state during an active trace (reset on each context enter/exit)
        self._active_trace: Optional[BeaconTrace] = None

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register_container(self, container_id: str, module: Container) -> None:
        """
        Attach a forward hook to a container.
        Call once per container at model init time — not inside the training loop.
        """
        if container_id in self._containers:
            raise ValueError(f"Container '{container_id}' already registered.")

        self._containers[container_id] = module

        def _hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            if self._active_trace is None:
                return
            strength = out.detach().abs().mean().item()
            if strength > self.threshold:
                self._active_trace.path.append((container_id, strength))
                module.usage_count += 1

        handle = module.register_forward_hook(_hook)
        self._hooks.append(handle)

    def register_containers(self, containers: dict[str, Container]) -> None:
        """Convenience wrapper for registering multiple containers at once."""
        for cid, module in containers.items():
            self.register_container(cid, module)

    # -----------------------------------------------------------------------
    # Trace context manager
    # -----------------------------------------------------------------------

    class _TraceContext:
        def __init__(self, tracer: "BeaconTracer", step: int):
            self._tracer = tracer
            self._step = step

        def __enter__(self) -> torch.Tensor:
            t = self._tracer
            beacon_id = uuid.uuid4().hex[:8]
            t._active_trace = BeaconTrace(beacon_id=beacon_id, step=self._step)
            # Unique low-magnitude vector. randn gives different direction each
            # time — important so we don't bias the same path repeatedly.
            beacon = torch.randn(t.dim) * t.magnitude
            return beacon

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            t = self._tracer
            if t._active_trace is not None:
                t._commit_trace(t._active_trace)
                t._active_trace = None
            return False  # don't suppress exceptions

    def trace(self, step: int) -> "_TraceContext":
        """
        Context manager for a single traced forward pass.

        Yields a beacon tensor to add to your input.
        On exit, commits the recorded path to the graph.
        """
        return self._TraceContext(self, step)

    # -----------------------------------------------------------------------
    # Graph updates
    # -----------------------------------------------------------------------

    def _commit_trace(self, trace: BeaconTrace) -> None:
        """Strengthen edges between every consecutive pair in the path."""
        self.traces.append(trace)
        path = trace.path
        for i in range(len(path) - 1):
            src, src_s = path[i]
            dst, dst_s = path[i + 1]
            self.path_graph[src][dst] += (src_s + dst_s) / 2.0

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    @property
    def last_trace(self) -> Optional[BeaconTrace]:
        return self.traces[-1] if self.traces else None

    def container_usage(self) -> dict[str, int]:
        """Return usage_count for every registered container."""
        return {cid: mod.usage_count for cid, mod in self._containers.items()}

    def least_used_containers(self, top_k: int = 5) -> list[tuple[str, int]]:
        """
        Containers with the lowest activation count.
        These are the best targets for routing new knowledge —
        high available capacity, low risk of overwriting existing paths.
        """
        usage = self.container_usage()
        return sorted(usage.items(), key=lambda x: x[1])[:top_k]

    def strongest_routes_to(self, container_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Which containers most reliably feed into container_id?"""
        incoming = {
            src: edges[container_id]
            for src, edges in self.path_graph.items()
            if container_id in edges
        }
        return sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def strongest_routes_from(self, container_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Which containers does container_id most reliably feed into?"""
        outgoing = self.path_graph.get(container_id, {})
        return sorted(outgoing.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def path_between(self, src: str, dst: str) -> float:
        """Direct edge weight between two containers. 0.0 if no direct connection."""
        return self.path_graph.get(src, {}).get(dst, 0.0)

    def subgraph(self, container_ids: list[str]) -> dict[str, dict[str, float]]:
        """Extract the induced subgraph for a subset of containers."""
        ids = set(container_ids)
        return {
            src: {dst: w for dst, w in edges.items() if dst in ids}
            for src, edges in self.path_graph.items()
            if src in ids
        }

    def hot_path(self, top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        The top_k strongest edges in the entire graph.
        These are the most load-bearing routes — the ones you most want
        to avoid overwriting when adding new knowledge.
        """
        edges = [
            (src, dst, w)
            for src, dsts in self.path_graph.items()
            for dst, w in dsts.items()
        ]
        return sorted(edges, key=lambda x: x[2], reverse=True)[:top_k]

    def suggest_insertion_point(self, top_k: int = 3) -> list[tuple[str, int]]:
        """
        Suggest the best containers to route new knowledge into.
        Combines low usage count (available capacity) with low in-graph
        centrality (low risk of disrupting existing paths).
        """
        usage = self.container_usage()
        # Simple score: lower is better — prefer unused containers with few connections
        in_degree = defaultdict(float)
        for _, edges in self.path_graph.items():
            for dst, w in edges.items():
                in_degree[dst] += w

        scores = {
            cid: usage.get(cid, 0) + in_degree.get(cid, 0.0)
            for cid in self._containers
        }
        return sorted(scores.items(), key=lambda x: x[1])[:top_k]

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Call this when done training to clean up forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __repr__(self):
        return (
            f"BeaconTracer("
            f"containers={len(self._containers)}, "
            f"traces={len(self.traces)}, "
            f"graph_edges={sum(len(v) for v in self.path_graph.values())})"
        )


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    DIM = 64
    N_CONTAINERS = 8

    # Build a small pool of containers
    containers = {f"slot_{i}": Container(dim=DIM, semantic_tag=f"slot_{i}") for i in range(N_CONTAINERS)}

    # Wire them into a trivial sequential model just to have a forward pass
    class ToyModel(nn.Module):
        def __init__(self, containers):
            super().__init__()
            self.slots = nn.ModuleDict(containers)
            self.head = nn.Linear(DIM, 10)

        def forward(self, x):
            for slot in self.slots.values():
                x = slot(x)
            return self.head(x)

    model = ToyModel(containers)
    tracer = BeaconTracer(dim=DIM)
    tracer.register_containers(containers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Running 20 training steps with beacon tracing...\n")

    for step in range(20):
        x = torch.randn(4, DIM)  # batch of 4
        y = torch.randint(0, 10, (4,))

        with tracer.trace(step=step) as beacon:
            # Beacon added to every item in the batch
            output = model(x + beacon.unsqueeze(0))

        loss = F.cross_entropy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:02d} | loss={loss.item():.4f} | {tracer.last_trace}")

    print("\n--- Path Graph (hot edges) ---")
    for src, dst, w in tracer.hot_path(top_k=5):
        print(f"  {src} → {dst}  weight={w:.4f}")

    print("\n--- Container Usage ---")
    for cid, count in sorted(tracer.container_usage().items(), key=lambda x: -x[1]):
        print(f"  {cid}: {count} activations")

    print("\n--- Best insertion points for new knowledge ---")
    for cid, score in tracer.suggest_insertion_point(top_k=3):
        print(f"  {cid}  (score={score:.2f})")

    print("\n--- Strongest routes INTO slot_3 ---")
    for src, w in tracer.strongest_routes_to("slot_3"):
        print(f"  {src} → slot_3  weight={w:.4f}")

    tracer.remove_hooks()
    print("\nDone.")
