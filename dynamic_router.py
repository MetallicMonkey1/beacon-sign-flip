"""
dynamic_router.py

The routing component of the Beacon-Routed Localist Memory (BRLM) architecture.

Role in the stack
-----------------
    Beacon Tracer   — builds the live path map during training        (done)
    Dynamic Router  — uses that map to decide which containers fire    (this)
    Container Pool  — growable set of localist knowledge slots         (next)

What the router does
--------------------
At inference (and during training), given an input vector the router:

    1. Computes affinity: query the input against every container's
       learned key vector. High affinity = "this container probably
       knows something relevant to this input."

    2. Applies the graph prior: containers that historically co-activate
       with high-affinity containers get a score boost. The beacon map
       is used as a structural prior, not just a diagnostic.

    3. Graph hop: one-hop neighborhood expansion along the strongest
       edges. If container A is highly relevant, containers that A
       typically leads to are partially activated too.

    4. Selects top-k containers by combined score (sparse).

    5. Runs the input through only those containers, weighted by
       their normalized scores.

    6. Returns the transformed output + routing metadata including
       an auxiliary load-balancing loss.

Differentiability
-----------------
Hard top-k selection is not differentiable. We handle this with a
straight-through estimator: hard selection in the forward pass,
soft scores in the backward pass (gradients flow through the full
softmax, not just the top-k subset). This is the same approach used
in Switch Transformer and other sparse MoE systems.

Load balancing
--------------
Without a regularizer, routers collapse: every input routes to the
same 2-3 containers and the rest never train. We add an auxiliary
load-balancing loss (following Switch Transformer) that penalizes
uneven container utilization. Add a small multiple of this to your
main loss (lambda ~1e-2 is usually enough).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from beacon_tracer import BeaconTracer, Container


# ---------------------------------------------------------------------------
# RouterOutput — everything the router produces in one object
# ---------------------------------------------------------------------------

@dataclass
class RouterOutput:
    """
    Everything produced by a single router forward pass.

    Fields
    ------
    output          Final transformed tensor (same shape as input).
    activated_ids   Ordered list of container IDs that fired.
    affinity_scores Raw (pre-graph) affinity of input to each container.
    combined_scores Final scores after graph prior applied (all containers).
    gate_weights    Normalized weights for the activated containers only.
    load_balance_loss  Auxiliary loss. Add scaled to main loss:
                       total_loss = task_loss + 0.01 * load_balance_loss
    """
    output: torch.Tensor
    activated_ids: list[str]
    affinity_scores: dict[str, float]
    combined_scores: dict[str, float]
    gate_weights: dict[str, float]
    load_balance_loss: torch.Tensor


# ---------------------------------------------------------------------------
# DynamicRouter
# ---------------------------------------------------------------------------

class DynamicRouter(nn.Module):
    """
    Sparse router that uses learned affinity + beacon path graph to
    select which containers process each input.

    Args
    ----
    dim             : Vector dimension. Must match Container.dim.
    container_ids   : Ordered list of container IDs to route between.
                      Must match the IDs registered with BeaconTracer.
    tracer          : Live BeaconTracer instance. Router reads
                      tracer.path_graph at every forward pass — the map
                      is always current, no need to rebuild.
    top_k           : Number of containers to activate per forward pass.
                      Rule of thumb: sqrt(n_containers) is a good start.
    graph_weight    : How much the beacon path graph influences routing.
                      0.0 = pure learned affinity.
                      1.0 = pure graph prior (ignores input content).
                      0.3 is a reasonable default.
    graph_hops      : How many hops to propagate from initially-selected
                      containers through the path graph. 1 is usually enough.
    temperature     : Softmax temperature for gate weight computation.
                      Lower = more winner-take-all. Higher = more distributed.
    """

    def __init__(
        self,
        dim: int,
        container_ids: list[str],
        tracer: "BeaconTracer",
        top_k: int = 3,
        graph_weight: float = 0.3,
        graph_hops: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()

        if top_k > len(container_ids):
            raise ValueError(f"top_k ({top_k}) cannot exceed number of containers ({len(container_ids)})")

        self.dim = dim
        self.container_ids = container_ids
        self.tracer = tracer
        self.top_k = top_k
        self.graph_weight = graph_weight
        self.graph_hops = graph_hops
        self.temperature = temperature

        # One learnable key per container.
        # Key = "the kind of input this container is specialized for."
        # Initialized small — the router starts neutral, specialization
        # emerges from training.
        self.keys = nn.ParameterDict({
            cid: nn.Parameter(torch.randn(dim) * 0.02)
            for cid in container_ids
        })

        # Project input into the same space as the keys.
        # Separate projection keeps routing logic independent of
        # the main representation stream.
        self.query_proj = nn.Linear(dim, dim, bias=False)

        # Running mean of container usage fraction for load balancing loss.
        # Not a parameter — updated in-place during training.
        self.register_buffer(
            "usage_ema",
            torch.ones(len(container_ids)) / len(container_ids),
        )
        self.ema_decay = 0.99

    # -----------------------------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------------------------

    def _normalized_graph_prior(self) -> torch.Tensor:
        """
        Convert tracer.path_graph into a per-container prior score vector.

        Score for container C = sum of incoming edge weights to C,
        normalized so the vector sums to 1. Containers with no incoming
        edges score 0 (they're leaf entry points — the input affinity
        term handles them).
        """
        in_weight = defaultdict(float)
        for _, edges in self.tracer.path_graph.items():
            for dst, w in edges.items():
                in_weight[dst] += w

        scores = torch.tensor(
            [in_weight.get(cid, 0.0) for cid in self.container_ids],
            dtype=torch.float32,
        )
        total = scores.sum()
        if total > 0:
            scores = scores / total
        return scores  # (n_containers,)

    def _graph_hop(
        self,
        seed_ids: list[str],
        all_scores: torch.Tensor,
        n_hops: int,
    ) -> torch.Tensor:
        """
        Propagate activation from seed containers through the path graph.

        For each hop, the score of each container is boosted by the
        weighted sum of its predecessors' current scores. This means
        containers that typically follow high-scoring containers inherit
        some of that score.

        Args
        ----
        seed_ids    : Container IDs selected in the first pass.
        all_scores  : Combined score tensor (n_containers,) before hopping.
        n_hops      : How many propagation steps.

        Returns
        -------
        Updated score tensor with graph-propagated boosts applied.
        """
        id_to_idx = {cid: i for i, cid in enumerate(self.container_ids)}
        scores = all_scores.clone()

        for _ in range(n_hops):
            delta = torch.zeros_like(scores)
            for src, edges in self.tracer.path_graph.items():
                if src not in id_to_idx:
                    continue
                src_idx = id_to_idx[src]
                src_score = scores[src_idx].item()
                if src_score == 0:
                    continue
                for dst, edge_w in edges.items():
                    if dst not in id_to_idx:
                        continue
                    dst_idx = id_to_idx[dst]
                    delta[dst_idx] += src_score * edge_w

            # Normalize delta and add as a small boost (don't overpower affinity)
            delta_norm = delta.sum()
            if delta_norm > 0:
                delta = delta / delta_norm * self.graph_weight
            scores = scores + delta

        return scores

    # -----------------------------------------------------------------------
    # Load balancing
    # -----------------------------------------------------------------------

    def _load_balance_loss(
        self,
        soft_scores: torch.Tensor,      # (batch, n_containers) full softmax
        selection_mask: torch.Tensor,   # (batch, n_containers) binary top-k mask
    ) -> torch.Tensor:
        """
        Switch Transformer-style auxiliary load balancing loss.

        Penalizes the product of:
          - mean router probability assigned to each container  (f_i)
          - fraction of tokens routed to each container         (P_i)

        Loss = n_containers * sum(f_i * P_i)

        When perfectly balanced both are 1/n, and loss = 1.
        Routing collapse makes this much larger.
        """
        n = len(self.container_ids)
        # f_i: mean soft probability for container i over batch
        f = soft_scores.mean(dim=0)                         # (n,)
        # P_i: fraction of examples that selected container i
        p = selection_mask.float().mean(dim=0)              # (n,)
        return n * (f * p).sum()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        containers: nn.ModuleDict,
    ) -> RouterOutput:
        """
        Route x through the top-k most relevant containers.

        Args
        ----
        x           : Input tensor. Shape (batch, dim) or (batch, seq, dim).
                      If 3D, routing decision is made on the mean over seq.
        containers  : nn.ModuleDict mapping container_id → Container module.
                      Only activated containers are called — the rest are skipped.

        Returns
        -------
        RouterOutput with transformed tensor and full routing metadata.
        """
        batch_shape = x.shape[:-1]
        seq_dim = x.dim() == 3

        # --- 1. Query projection ------------------------------------------
        # Pool to (batch, dim) for the routing decision
        x_pool = x.mean(dim=1) if seq_dim else x          # (batch, dim)
        query = self.query_proj(x_pool)                    # (batch, dim)

        # --- 2. Affinity scores -------------------------------------------
        # Dot product of query against each container's key, scaled
        key_matrix = torch.stack(
            [self.keys[cid] for cid in self.container_ids], dim=0
        )                                                   # (n, dim)
        affinity = (query @ key_matrix.T) / (self.dim ** 0.5)  # (batch, n)

        # Soft scores for the full batch (used in load balance loss + backward)
        soft_scores = F.softmax(affinity / self.temperature, dim=-1)  # (batch, n)

        # Mean affinity over batch for routing decision (single routing per batch)
        mean_affinity = affinity.mean(dim=0)               # (n,)

        # --- 3. Graph prior -----------------------------------------------
        graph_prior = self._normalized_graph_prior().to(x.device)  # (n,)

        combined = (
            (1.0 - self.graph_weight) * F.softmax(mean_affinity, dim=0)
            + self.graph_weight * graph_prior
        )                                                   # (n,)

        # --- 4. Graph hop -------------------------------------------------
        if self.graph_hops > 0 and len(self.tracer.path_graph) > 0:
            # Seed with the current top-k before hopping
            _, seed_indices = combined.topk(min(self.top_k, combined.size(0)))
            seed_ids = [self.container_ids[i] for i in seed_indices.tolist()]
            combined = self._graph_hop(seed_ids, combined, self.graph_hops)

        # --- 5. Top-k selection -------------------------------------------
        top_k = min(self.top_k, len(self.container_ids))
        top_vals, top_indices = combined.topk(top_k)
        activated_ids = [self.container_ids[i] for i in top_indices.tolist()]

        # Binary selection mask for load balance loss
        selection_mask = torch.zeros(
            x_pool.size(0), len(self.container_ids), device=x.device
        )
        selection_mask[:, top_indices] = 1.0              # (batch, n)

        # --- 6. Gate weights (straight-through) ---------------------------
        # Forward: hard normalized weights for selected containers
        gate_weights_hard = F.softmax(top_vals / self.temperature, dim=0)  # (top_k,)

        # Backward: gradients flow through the full soft_scores tensor.
        # We detach the hard weights and add back the soft path for gradients.
        # This is the straight-through estimator.
        soft_selected = soft_scores[:, top_indices].mean(dim=0)  # (top_k,) soft
        gate_weights = (
            gate_weights_hard.detach()
            + soft_selected
            - soft_selected.detach()
        )                                                  # hard forward, soft backward

        # --- 7. Apply containers ------------------------------------------
        output = x.clone()
        for cid, weight in zip(activated_ids, gate_weights):
            container_out = containers[cid](x)             # same shape as x
            output = output + weight * container_out       # weighted residual

        # --- 8. Update usage EMA (for monitoring, not training) -----------
        with torch.no_grad():
            usage_this_step = selection_mask.mean(dim=0)  # (n,)
            self.usage_ema = (
                self.ema_decay * self.usage_ema
                + (1 - self.ema_decay) * usage_this_step
            )

        # --- 9. Load balancing loss ---------------------------------------
        lb_loss = self._load_balance_loss(soft_scores, selection_mask)

        # --- 10. Build metadata dictionaries ------------------------------
        affinity_dict = {
            cid: mean_affinity[i].item()
            for i, cid in enumerate(self.container_ids)
        }
        combined_dict = {
            cid: combined[i].item()
            for i, cid in enumerate(self.container_ids)
        }
        gate_dict = {
            cid: gate_weights[j].item()
            for j, cid in enumerate(activated_ids)
        }

        return RouterOutput(
            output=output,
            activated_ids=activated_ids,
            affinity_scores=affinity_dict,
            combined_scores=combined_dict,
            gate_weights=gate_dict,
            load_balance_loss=lb_loss,
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def routing_distribution(self) -> dict[str, float]:
        """
        Current EMA routing distribution across all containers.
        Values sum to ~top_k/n_containers when balanced.
        Use this to spot routing collapse during training.
        """
        return {
            cid: self.usage_ema[i].item()
            for i, cid in enumerate(self.container_ids)
        }

    def is_collapsed(self, threshold: float = 0.8) -> bool:
        """
        Returns True if top_k containers are absorbing threshold fraction
        of all routing traffic — sign of routing collapse.
        """
        top_usage = sorted(self.usage_ema.tolist(), reverse=True)[:self.top_k]
        return sum(top_usage) > threshold

    def __repr__(self):
        return (
            f"DynamicRouter("
            f"containers={len(self.container_ids)}, "
            f"top_k={self.top_k}, "
            f"graph_weight={self.graph_weight}, "
            f"collapsed={self.is_collapsed()})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from beacon_tracer import Container, BeaconTracer

    torch.manual_seed(42)

    DIM = 64
    N_CONTAINERS = 8
    TOP_K = 3
    BATCH = 4

    # --- Build container pool ---------------------------------------------
    containers = nn.ModuleDict({
        f"slot_{i}": Container(dim=DIM, semantic_tag=f"slot_{i}")
        for i in range(N_CONTAINERS)
    })
    container_ids = list(containers.keys())

    # --- Build tracer and run a few steps to populate the path graph ------
    tracer = BeaconTracer(dim=DIM)
    tracer.register_containers({cid: containers[cid] for cid in container_ids})

    # Simple sequential model to generate a populated path graph
    class ToySeqModel(nn.Module):
        def __init__(self, containers):
            super().__init__()
            self.slots = containers
        def forward(self, x):
            for slot in self.slots.values():
                x = slot(x)
            return x

    seq_model = ToySeqModel(containers)
    print("Warming up beacon tracer (20 steps)...")
    for step in range(20):
        x = torch.randn(BATCH, DIM)
        with tracer.trace(step=step) as beacon:
            _ = seq_model(x + beacon.unsqueeze(0))

    print(f"Path graph populated: {sum(len(v) for v in tracer.path_graph.values())} edges\n")

    # --- Build router ------------------------------------------------------
    router = DynamicRouter(
        dim=DIM,
        container_ids=container_ids,
        tracer=tracer,
        top_k=TOP_K,
        graph_weight=0.3,
        graph_hops=1,
        temperature=1.0,
    )
    print(router)

    # --- Head and optimizer ------------------------------------------------
    head = nn.Linear(DIM, 10)
    optimizer = torch.optim.Adam(
        list(router.parameters()) + list(containers.parameters()) + list(head.parameters()),
        lr=1e-3,
    )

    # --- Training loop with router -----------------------------------------
    print("\nTraining with router (20 steps):")
    print(f"{'Step':>4}  {'Task Loss':>10}  {'LB Loss':>10}  {'Activated containers'}")
    print("-" * 65)

    for step in range(20):
        x = torch.randn(BATCH, DIM)
        y = torch.randint(0, 10, (BATCH,))

        result: RouterOutput = router(x, containers)

        logits = head(result.output)
        task_loss = F.cross_entropy(logits, y)
        total_loss = task_loss + 0.01 * result.load_balance_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            activated = ", ".join(result.activated_ids)
            print(
                f"{step:>4}  {task_loss.item():>10.4f}  "
                f"{result.load_balance_loss.item():>10.4f}  [{activated}]"
            )

    # --- Introspection -----------------------------------------------------
    print("\n--- Routing distribution (EMA) ---")
    for cid, usage in sorted(router.routing_distribution().items(), key=lambda x: -x[1]):
        bar = "█" * int(usage * 200)
        print(f"  {cid}: {usage:.4f}  {bar}")

    print(f"\nRouting collapsed: {router.is_collapsed()}")

    print("\n--- Gate weights for last forward pass ---")
    for cid, w in sorted(result.gate_weights.items(), key=lambda x: -x[1]):
        print(f"  {cid}: {w:.4f}")

    print("\n--- Top combined scores (pre-activation) ---")
    top_scores = sorted(result.combined_scores.items(), key=lambda x: -x[1])[:5]
    for cid, score in top_scores:
        marker = " ← activated" if cid in result.activated_ids else ""
        print(f"  {cid}: {score:.4f}{marker}")

    tracer.remove_hooks()
    print("\nDone.")
