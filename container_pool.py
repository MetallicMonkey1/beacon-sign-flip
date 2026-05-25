"""
container_pool.py

The ContainerPool component of the Beacon-Routed Localist Memory (BRLM) architecture.

Role in the stack
-----------------
    BeaconTracer    — builds the live path map during training           (done)
    DynamicRouter   — uses that map to select which containers fire      (done)
    ContainerPool   — growable set of slots + model mapping + grafting   (this)

The key capability this file adds
----------------------------------
An existing pretrained model (GPT-2, LLaMA, BERT, any transformer or
arbitrary nn.Module) can be MAPPED — the beacon tracer hooks into its
internal layers, builds a path graph of how activations flow through it,
then new containers are GRAFTED at a chosen point in that graph.

From that point forward:
    - The existing weights are UNTOUCHED and FROZEN
    - New knowledge routes to new containers
    - The beacon map grows to include both old layers and new containers
    - You can add more containers at any time — each one registers
      itself automatically with tracer and router

Three classes
-------------
    ModelMapper     — scans any nn.Module, registers its layers as
                      virtual nodes in the tracer, runs calibration
                      passes to populate the path graph

    ContainerPool   — the growable container set. Handles lifecycle:
                      add, remove, auto-register with tracer + router

    GraftedModel    — wraps an existing model with a ContainerPool
                      injected at a named layer via a forward hook.
                      The base model runs normally; the hook intercepts
                      activations at the graft point, passes them through
                      the router + pool, and lets them continue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings

from beacon_tracer import BeaconTracer, Container
from dynamic_router import DynamicRouter, RouterOutput


# ---------------------------------------------------------------------------
# ModelMapper
# ---------------------------------------------------------------------------

class ModelMapper:
    """
    Scans an existing pretrained model and registers its internal layers
    as virtual nodes in a BeaconTracer path graph.

    "Virtual" means: the tracer hooks into real existing layers and tracks
    their activations as path nodes — but those layers have no Container
    object, no learnable keys, and are never modified. They're read-only
    landmarks in the map.

    After mapping, you know:
        - Which layers activate for typical inputs
        - How strongly they activate (average magnitude)
        - Which layers reliably co-activate (path graph edges)
        - Where capacity is low → good graft points for new containers

    Typical usage
    -------------
        mapper = ModelMapper(tracer, dim=768)
        mapper.map(pretrained_model, depth=2)
        mapper.calibrate(pretrained_model, calibration_inputs, n_steps=50)
        graft_points = mapper.suggest_graft_points(top_k=3)
    """

    def __init__(self, tracer: BeaconTracer, dim: int):
        self.tracer = tracer
        self.dim = dim
        self._hooks: list = []
        self._virtual_nodes: dict[str, nn.Module] = {}  # name → module
        self._activation_stats: dict[str, list[float]] = defaultdict(list)

    # -----------------------------------------------------------------------
    # Mapping
    # -----------------------------------------------------------------------

    def map(
        self,
        model: nn.Module,
        depth: int = 2,
        include_types: tuple = (nn.Linear, nn.LayerNorm, nn.MultiheadAttention),
        name_prefix: str = "existing",
    ) -> dict[str, nn.Module]:
        """
        Recursively scan model and register matching layers as virtual nodes.

        Args
        ----
        model         : Any pretrained nn.Module.
        depth         : How many levels deep to scan. depth=2 gives
                        top-level blocks; depth=4 gives individual sublayers.
        include_types : Only register layers of these types. Default covers
                        the core transformer primitives.
        name_prefix   : Prefix for node IDs in the path graph.
                        e.g. "existing.transformer.h.0.attn"

        Returns
        -------
        Dict of node_id → module for all registered virtual nodes.
        """
        self._scan(model, name_prefix, depth, current_depth=0, include_types=include_types)
        print(f"[ModelMapper] Registered {len(self._virtual_nodes)} virtual nodes from existing model.")
        return dict(self._virtual_nodes)

    def _scan(
        self,
        module: nn.Module,
        prefix: str,
        max_depth: int,
        current_depth: int,
        include_types: tuple,
    ) -> None:
        if current_depth > max_depth:
            return

        for name, child in module.named_children():
            node_id = f"{prefix}.{name}"

            if isinstance(child, include_types):
                self._register_virtual_node(node_id, child)

            # Recurse regardless of whether we registered this level
            self._scan(child, node_id, max_depth, current_depth + 1, include_types)

    def _register_virtual_node(self, node_id: str, module: nn.Module) -> None:
        """Attach a forward hook to an existing layer, log it as a path node."""
        if node_id in self._virtual_nodes:
            return

        self._virtual_nodes[node_id] = module

        # We cannot use tracer.register_container here (that needs a Container object).
        # Instead we directly attach a hook that writes into the tracer's active trace.
        def _hook(mod, inp, out):
            if self.tracer._active_trace is None:
                return
            # Normalize output — handles scalars, 2D, 3D tensors
            if isinstance(out, torch.Tensor):
                strength = out.detach().float().abs().mean().item()
            elif isinstance(out, (tuple, list)):
                # e.g. MultiheadAttention returns (output, attn_weights)
                strength = out[0].detach().float().abs().mean().item()
            else:
                return

            self._activation_stats[node_id].append(strength)

            if strength > self.tracer.threshold:
                self.tracer._active_trace.path.append((node_id, strength))

        handle = module.register_forward_hook(_hook)
        self._hooks.append(handle)

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        inputs: list[torch.Tensor],
        n_steps: int = 50,
        forward_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Run the beacon tracer through the existing model to populate
        the path graph before any containers are added.

        Args
        ----
        model         : The pretrained model to trace.
        inputs        : List of representative input tensors. Will be
                        cycled if len(inputs) < n_steps.
        n_steps       : Number of forward passes to run.
        forward_kwargs: Extra kwargs to pass to model.forward()
                        (e.g. attention_mask for transformers).
        """
        forward_kwargs = forward_kwargs or {}
        model.eval()

        print(f"[ModelMapper] Calibrating over {n_steps} steps...")

        with torch.no_grad():
            for step in range(n_steps):
                x = inputs[step % len(inputs)]

                # Beacon needs to match input shape / be injectable.
                # We inject by adding to the first dimension of x.
                beacon_vec = torch.randn(self.dim, device=x.device) * self.tracer.magnitude

                with self.tracer.trace(step=step) as beacon:
                    # For flat inputs: add beacon to each item in batch
                    if x.dim() == 2:
                        x_beaconed = x + beacon.unsqueeze(0)
                    elif x.dim() == 3:
                        x_beaconed = x + beacon.unsqueeze(0).unsqueeze(0)
                    else:
                        x_beaconed = x

                    try:
                        _ = model(x_beaconed, **forward_kwargs)
                    except Exception:
                        # Some models are strict about input types.
                        # Fall back to unbeaconed forward — path graph still
                        # builds from the hooks, beacon just won't be in input.
                        _ = model(x, **forward_kwargs)

        n_edges = sum(len(v) for v in self.tracer.path_graph.values())
        print(f"[ModelMapper] Calibration complete. Path graph: {n_edges} edges, "
              f"{len(self.tracer.traces)} traces recorded.")

    # -----------------------------------------------------------------------
    # Graft point suggestions
    # -----------------------------------------------------------------------

    def suggest_graft_points(self, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Suggest the best layers in the existing model to graft containers onto.

        A good graft point has:
        - High average activation (it's on a live path, so containers here
          will be fed real signal)
        - Moderate incoming graph weight (not so overloaded that grafting
          would disrupt critical paths)

        Returns list of (node_id, score) sorted by suitability.
        """
        # Mean activation strength per node
        mean_activation = {
            node_id: sum(vals) / len(vals)
            for node_id, vals in self._activation_stats.items()
            if vals
        }

        # Incoming graph weight per node (centrality)
        in_weight = defaultdict(float)
        for _, edges in self.tracer.path_graph.items():
            for dst, w in edges.items():
                in_weight[dst] += w

        # Score: high activation, low centrality (less risk of disruption)
        scores = {}
        for node_id, act in mean_activation.items():
            centrality = in_weight.get(node_id, 0.0)
            # Normalize centrality by max to put on same scale as activation
            scores[node_id] = act / (1.0 + centrality)

        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    def activation_stats(self) -> dict[str, dict]:
        """Summary statistics for each mapped layer."""
        result = {}
        for node_id, vals in self._activation_stats.items():
            if vals:
                t = torch.tensor(vals)
                result[node_id] = {
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "min": t.min().item(),
                    "max": t.max().item(),
                    "n_samples": len(vals),
                }
        return result

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# ContainerPool
# ---------------------------------------------------------------------------

class ContainerPool(nn.Module):
    """
    Growable pool of localist knowledge containers.

    This is the single source of truth for the container set.
    The router and tracer both read from it — adding a container here
    automatically propagates to both.

    Key properties
    --------------
    - Add containers at any time, including after training has started
    - New containers start empty (random init, near-zero output)
    - Beacon naturally discovers and populates them as capacity pressure builds
    - Remove containers without touching others
    - Works for both BRLM-native models and grafted-onto-existing models

    Args
    ----
    dim             : Container vector dimension.
    tracer          : BeaconTracer instance (shared with router).
    router          : DynamicRouter instance. Pool keeps it in sync
                      when containers are added/removed.
    hidden_dim_ratio: Container internal MLP width = dim * ratio.
    """

    def __init__(
        self,
        dim: int,
        tracer: BeaconTracer,
        router: Optional[DynamicRouter] = None,
        hidden_dim_ratio: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.tracer = tracer
        self.router = router
        self.hidden_dim_ratio = hidden_dim_ratio

        # The actual containers — nn.ModuleDict so PyTorch tracks parameters
        self.containers: nn.ModuleDict = nn.ModuleDict()

        # Metadata that lives outside the module (not tensors)
        self._container_meta: dict[str, dict] = {}
        self._next_id: int = 0

    # -----------------------------------------------------------------------
    # Container lifecycle
    # -----------------------------------------------------------------------

    def add_container(
        self,
        semantic_tag: str = "",
        container_id: Optional[str] = None,
    ) -> str:
        """
        Allocate a new empty container and register it with tracer + router.

        Args
        ----
        semantic_tag  : Optional human-readable label. Can be set now or
                        later once the beacon reveals what lives here.
        container_id  : Optional explicit ID. Auto-generated if not given.

        Returns
        -------
        The container_id of the newly created container.
        """
        if container_id is None:
            container_id = f"container_{self._next_id:04d}"
            self._next_id += 1

        if container_id in self.containers:
            raise ValueError(f"Container '{container_id}' already exists.")

        # Create container with near-zero output init so it starts neutral
        c = Container(
            dim=self.dim,
            hidden_dim=self.dim * self.hidden_dim_ratio,
            semantic_tag=semantic_tag,
        )
        # Zero the output projection so new container starts as identity
        # (doesn't corrupt existing behavior immediately on graft)
        with torch.no_grad():
            c.net[-1].weight.zero_()
            c.net[-1].bias.zero_()

        self.containers[container_id] = c
        self._container_meta[container_id] = {
            "semantic_tag": semantic_tag,
            "added_at_step": len(self.tracer.traces),
        }

        # Register with tracer
        self.tracer.register_container(container_id, c)

        # Register with router (dynamic grow)
        if self.router is not None:
            self._grow_router(container_id)

        print(f"[ContainerPool] Added '{container_id}'"
              + (f" [{semantic_tag}]" if semantic_tag else "")
              + f"  (pool size: {len(self.containers)})")

        return container_id

    def remove_container(self, container_id: str) -> None:
        """
        Remove a container from the pool.

        The path graph retains historical edges to/from this container
        (useful for forensics) but the router will no longer route to it.
        """
        if container_id not in self.containers:
            raise KeyError(f"Container '{container_id}' not found.")

        del self.containers[container_id]
        del self._container_meta[container_id]

        # Remove from router
        if self.router is not None and container_id in self.router.container_ids:
            self.router.container_ids.remove(container_id)
            if container_id in self.router.keys:
                del self.router.keys[container_id]
            # Rebuild usage_ema without this container
            ids = self.router.container_ids
            self.router.register_buffer(
                "usage_ema",
                torch.ones(len(ids)) / max(len(ids), 1)
            )

        print(f"[ContainerPool] Removed '{container_id}'  (pool size: {len(self.containers)})")

    def expand(self, n: int, tag_prefix: str = "") -> list[str]:
        """
        Add n new empty containers at once.
        Uses tracer's suggest_insertion_point to name them informatively.

        Returns list of new container IDs.
        """
        new_ids = []
        for i in range(n):
            tag = f"{tag_prefix}_{i}" if tag_prefix else ""
            cid = self.add_container(semantic_tag=tag)
            new_ids.append(cid)
        return new_ids

    def expand_if_pressured(
        self,
        pressure_threshold: float = 0.8,
        add_n: int = 4,
    ) -> list[str]:
        """
        Auto-expand when existing containers are over-utilized.

        Pressure = fraction of containers above 80% of max usage.
        If pressure > threshold, add add_n new containers.

        Call this periodically during training (e.g. every 1000 steps).
        """
        if not self.containers:
            return []

        usages = [c.usage_count for c in self.containers.values()]
        max_usage = max(usages) if usages else 1
        if max_usage == 0:
            return []

        pressure = sum(u / max_usage > 0.8 for u in usages) / len(usages)

        if pressure > pressure_threshold:
            print(f"[ContainerPool] Capacity pressure {pressure:.1%} > {pressure_threshold:.1%} "
                  f"— expanding by {add_n} containers.")
            return self.expand(add_n)

        return []

    # -----------------------------------------------------------------------
    # Router sync
    # -----------------------------------------------------------------------

    def _grow_router(self, container_id: str) -> None:
        """Add a new container to the router without rebuilding it."""
        router = self.router

        router.container_ids.append(container_id)

        # Add new learnable key (starts near zero — neutral routing)
        router.keys[container_id] = nn.Parameter(
            torch.randn(self.dim, device=next(router.parameters()).device) * 0.02
        )

        # Grow usage_ema buffer
        new_ema = torch.cat([
            router.usage_ema,
            torch.tensor([1.0 / len(router.container_ids)])
        ])
        router.register_buffer("usage_ema", new_ema)

    def attach_router(self, router: DynamicRouter) -> None:
        """
        Attach a router after pool construction, and sync all existing
        containers into it. Useful when pool is built before router.
        """
        self.router = router
        for cid in self.containers:
            if cid not in router.container_ids:
                self._grow_router(cid)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @property
    def container_ids(self) -> list[str]:
        return list(self.containers.keys())

    @property
    def size(self) -> int:
        return len(self.containers)

    def usage_report(self) -> dict[str, dict]:
        """Per-container usage stats + metadata."""
        report = {}
        for cid, c in self.containers.items():
            meta = self._container_meta.get(cid, {})
            report[cid] = {
                "semantic_tag": c.semantic_tag,
                "usage_count": c.usage_count,
                "added_at_step": meta.get("added_at_step", 0),
                "in_graph_weight": sum(
                    edges.get(cid, 0.0)
                    for edges in self.tracer.path_graph.values()
                ),
            }
        return report

    def cold_containers(self, threshold: int = 5) -> list[str]:
        """Containers that have been activated fewer than threshold times."""
        return [
            cid for cid, c in self.containers.items()
            if c.usage_count < threshold
        ]

    def hot_containers(self, top_k: int = 5) -> list[tuple[str, int]]:
        """The most heavily used containers."""
        usage = [(cid, c.usage_count) for cid, c in self.containers.items()]
        return sorted(usage, key=lambda x: -x[1])[:top_k]

    def __repr__(self):
        return (
            f"ContainerPool("
            f"size={self.size}, "
            f"dim={self.dim}, "
            f"total_activations={sum(c.usage_count for c in self.containers.values())})"
        )


# ---------------------------------------------------------------------------
# GraftedModel
# ---------------------------------------------------------------------------

class GraftedModel(nn.Module):
    """
    Wraps an existing pretrained model with a ContainerPool grafted at
    a named layer.

    How it works
    ------------
    A forward hook is registered on the target layer. When the base model
    runs normally and reaches that layer, the hook:
        1. Captures the layer's output tensor
        2. Passes it through the router → container pool
        3. Returns the enriched tensor
        4. The base model continues with the enriched activation

    The base model is NEVER modified. Its weights can be frozen.
    The only new trainable parameters are the container pool + router keys.

    Args
    ----
    base_model      : Any pretrained nn.Module.
    pool            : ContainerPool to graft on.
    router          : DynamicRouter to use for routing at the graft point.
    graft_layer     : The nn.Module instance to hook into. Get this via
                      base_model.transformer.h[6] or similar.
    graft_layer_name: Human-readable name for logging.
    freeze_base     : If True, freeze all base model parameters.
                      New containers still train. Default True.
    """

    def __init__(
        self,
        base_model: nn.Module,
        pool: ContainerPool,
        router: DynamicRouter,
        graft_layer: nn.Module,
        graft_layer_name: str = "graft_point",
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.pool = pool
        self.router = router
        self.graft_layer_name = graft_layer_name

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            print(f"[GraftedModel] Base model frozen. "
                  f"Trainable params: container pool + router keys only.")

        # Track last routing result for inspection
        self._last_router_output: Optional[RouterOutput] = None

        # Register the graft hook
        self._graft_hook = graft_layer.register_forward_hook(self._hook_fn)
        print(f"[GraftedModel] Grafted onto '{graft_layer_name}'.")

    def _hook_fn(
        self,
        module: nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Intercept output of graft layer, pass through container pool,
        return enriched activation.
        """
        # Handle tuple outputs (e.g. attention layers return (out, weights))
        if isinstance(out, tuple):
            tensor_out, *rest = out
        else:
            tensor_out = out
            rest = []

        result: RouterOutput = self.router(tensor_out, self.pool.containers)
        self._last_router_output = result

        enriched = result.output

        if rest:
            return (enriched, *rest)
        return enriched

    def forward(self, *args, **kwargs):
        """
        Pass through to base model. The graft hook fires automatically
        at the registered layer.
        """
        return self.base_model(*args, **kwargs)

    @property
    def last_routing(self) -> Optional[RouterOutput]:
        """Inspect which containers fired on the last forward pass."""
        return self._last_router_output

    def add_containers(self, n: int, tag_prefix: str = "") -> list[str]:
        """Convenience: expand pool and immediately use in next forward pass."""
        return self.pool.expand(n, tag_prefix=tag_prefix)

    def remove_graft(self) -> None:
        """Remove the hook and restore base model to its original behavior."""
        self._graft_hook.remove()
        print(f"[GraftedModel] Graft removed from '{self.graft_layer_name}'.")

    def trainable_parameters(self):
        """Returns only the new parameters (pool + router), not base model."""
        return (
            list(self.pool.parameters())
            + list(self.router.parameters())
        )

    def __repr__(self):
        n_base = sum(p.numel() for p in self.base_model.parameters())
        n_new = sum(p.numel() for p in self.trainable_parameters())
        return (
            f"GraftedModel(\n"
            f"  base_model params : {n_base:,}\n"
            f"  new params (pool+router): {n_new:,}  "
            f"({100*n_new/(n_base+n_new):.2f}% of total)\n"
            f"  graft_point: '{self.graft_layer_name}'\n"
            f"  {self.pool}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from beacon_tracer import BeaconTracer, Container
    from dynamic_router import DynamicRouter

    torch.manual_seed(42)

    DIM = 64
    BATCH = 4

    # -----------------------------------------------------------------------
    # 1. Simulate a pretrained model (two transformer-style blocks)
    # -----------------------------------------------------------------------

    class FakeTransformerBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        def forward(self, x):
            x = self.norm(x + self.attn(x))
            x = self.norm(x + self.mlp(x))
            return x

    class FakePretrainedModel(nn.Module):
        def __init__(self, dim, n_blocks=4):
            super().__init__()
            self.blocks = nn.ModuleList([FakeTransformerBlock(dim) for _ in range(n_blocks)])
            self.head = nn.Linear(dim, 10)
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    pretrained = FakePretrainedModel(dim=DIM, n_blocks=4)
    print("=" * 60)
    print("PRETRAINED MODEL")
    print(f"  Params: {sum(p.numel() for p in pretrained.parameters()):,}")

    # -----------------------------------------------------------------------
    # 2. Map the pretrained model
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("MAPPING EXISTING MODEL")

    tracer = BeaconTracer(dim=DIM)
    mapper = ModelMapper(tracer=tracer, dim=DIM)

    # Register existing layers as virtual path nodes
    virtual_nodes = mapper.map(pretrained, depth=3)

    # Calibrate: run beacon traces through existing model
    calibration_inputs = [torch.randn(BATCH, DIM) for _ in range(10)]
    mapper.calibrate(pretrained, calibration_inputs, n_steps=30)

    # See what the map found
    print("\n--- Suggested graft points ---")
    for node_id, score in mapper.suggest_graft_points(top_k=3):
        print(f"  {node_id}  (score={score:.4f})")

    # -----------------------------------------------------------------------
    # 3. Build pool + router (initially empty)
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING CONTAINER POOL")

    pool = ContainerPool(dim=DIM, tracer=tracer)

    # Seed with a few containers
    pool.add_container(semantic_tag="general_A")
    pool.add_container(semantic_tag="general_B")
    pool.add_container(semantic_tag="general_C")

    # Router over current pool
    router = DynamicRouter(
        dim=DIM,
        container_ids=pool.container_ids,
        tracer=tracer,
        top_k=2,
        graph_weight=0.2,
    )
    pool.attach_router(router)

    # -----------------------------------------------------------------------
    # 4. Graft onto the existing model at block[2]
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAFTING")

    grafted = GraftedModel(
        base_model=pretrained,
        pool=pool,
        router=router,
        graft_layer=pretrained.blocks[2],
        graft_layer_name="blocks.2",
        freeze_base=True,
    )
    print(grafted)

    # -----------------------------------------------------------------------
    # 5. Train only the new containers
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING (new containers only, base model frozen)")
    print(f"{'Step':>4}  {'Loss':>8}  {'LB Loss':>8}  {'Activated'}")
    print("-" * 55)

    optimizer = torch.optim.Adam(grafted.trainable_parameters(), lr=1e-3)

    for step in range(20):
        x = torch.randn(BATCH, DIM)
        y = torch.randint(0, 10, (BATCH,))

        with tracer.trace(step=step + 30) as beacon:
            out = grafted(x + beacon.unsqueeze(0))

        loss = F.cross_entropy(out, y)
        lb_loss = grafted.last_routing.load_balance_loss
        total_loss = loss + 0.01 * lb_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            activated = ", ".join(grafted.last_routing.activated_ids)
            print(f"{step:>4}  {loss.item():>8.4f}  {lb_loss.item():>8.4f}  [{activated}]")

    # -----------------------------------------------------------------------
    # 6. Add more containers mid-training (the whole point)
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("ADDING 3 NEW CONTAINERS MID-TRAINING")
    new_ids = grafted.add_containers(3, tag_prefix="expansion")
    print(f"New containers: {new_ids}")

    # Continue training — new containers start cold, get routed to gradually
    print("\nContinuing training with expanded pool:")
    # Only add the NEW containers' parameters — existing ones already in optimizer
    new_params = [p for cid in new_ids for p in pool.containers[cid].parameters()]
    new_params += [pool.router.keys[cid] for cid in new_ids]
    optimizer.add_param_group({"params": new_params})

    for step in range(20, 35):
        x = torch.randn(BATCH, DIM)
        y = torch.randint(0, 10, (BATCH,))

        with tracer.trace(step=step + 30) as beacon:
            out = grafted(x + beacon.unsqueeze(0))

        loss = F.cross_entropy(out, y)
        lb_loss = grafted.last_routing.load_balance_loss
        total = loss + 0.01 * lb_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step % 5 == 0:
            activated = ", ".join(grafted.last_routing.activated_ids)
            print(f"{step:>4}  {loss.item():>8.4f}  {lb_loss.item():>8.4f}  [{activated}]")

    # -----------------------------------------------------------------------
    # 7. Final report
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL USAGE REPORT")
    for cid, stats in pool.usage_report().items():
        tag = f"[{stats['semantic_tag']}]" if stats['semantic_tag'] else ""
        print(f"  {cid} {tag:20s}  activations={stats['usage_count']:4d}  "
              f"graph_weight={stats['in_graph_weight']:.3f}  "
              f"added_at_step={stats['added_at_step']}")

    print(f"\nCold containers (< 5 activations): {pool.cold_containers(threshold=5)}")
    print(f"Hot containers: {pool.hot_containers(top_k=3)}")

    print(f"\nRouter collapsed: {router.is_collapsed()}")

    mapper.remove_hooks()
    tracer.remove_hooks()
    print("\nDone.")
