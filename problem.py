from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    cost: int
    resources: tuple[int, ...]


@dataclass
class Instance:
    name: str
    n: int
    k: int
    m: int
    node_resources: dict[int, tuple[int, ...]]
    edges: dict[tuple[int, int], Edge]


@dataclass
class Validation:
    valid: bool
    errors: list[str]
    cost: int
    available: tuple[int, ...]
    consumed: tuple[int, ...]
    slack: tuple[int, ...]


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))  ##ordem nao interessa 


##ler instancia
def read_instance(path: str | Path) -> Instance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()
             if line.strip()]
    header = tuple(map(int, lines[0].split()))
    n, edge_count, k, m=header
    
    nodes = {}
    for v, line in enumerate(lines[2:n + 2], start=1):
        resources = tuple(map(int, line.split()))
        nodes[v] = resources

    edges = {}
    for line in lines[n + 3:]:
        values = tuple(map(int, line.split()))
        u, v, cost, *resources = values
        key = edge_key(u, v)
        edges[key] = Edge(u, v, cost, tuple(resources))
    return Instance(path.stem, n, k, m, nodes, edges)


#ver se solução é valida
def validate_solution(instance: Instance, selected_edges: list[tuple[int, int]]) -> Validation:
    errors = []
    keys = [edge_key(u, v) for u, v in selected_edges]
    if len(keys) != instance.k - 1:
        errors.append(f"We need {instance.k - 1} edges.")
    if len(set(keys)) != len(keys):
        errors.append("Edges are repeated.")
    unknown = set(keys) - instance.edges.keys()
    if unknown:
        errors.append(f"Non existent edges: {sorted(unknown)}.")

    known = set(keys) & instance.edges.keys()
    nodes = {v for key in known for v in key}
    if len(nodes) != instance.k:
        errors.append(f"We need {instance.k} nodes.")

    adjacency = {v: [] for v in nodes}
    for u, v in known:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = set()
    components = 0
    for root in nodes:
        if root in seen:
            continue
        components += 1
        stack = [root]
        seen.add(root)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    if components != 1:
        errors.append("The nods dont form a connected structure.")
    if len(known) > len(nodes) - components:
        errors.append("Solution has cycles.")

    available = tuple(sum(instance.node_resources[v][l] for v in nodes)
                      for l in range(instance.m))
    consumed = tuple(sum(instance.edges[key].resources[l] for key in known)
                     for l in range(instance.m))
    slack = tuple(a - c for a, c in zip(available, consumed))
    for l, value in enumerate(slack, start=1):
        if value < 0:
            errors.append(f"Recurso {l}: faltam {-value} unidades.")
    cost = sum(instance.edges[key].cost for key in known)
    return Validation(not errors, errors, cost, available, consumed, slack)
