
from dataclasses import dataclass
from pathlib import Path
import argparse
import random

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

#####1 - contruction heurisitic
def construct_heu(instance: Instance, start:int, lvalue:float=8, normalization:str = "node", alpha: float | None = None, rng: random.Random | None = None):
    selected_nodes = {start}
    selected_edges = []
    slack = list(instance.node_resources[start])
    total_cost = 0
    if not instance.edges:
        return None

    ####calculo normalizado para depois comparar na pontuação, ver se ha melhor alternativa
    cost_reference = (sum(edge.cost for edge in instance.edges.values()) / len(instance.edges)) ##media
    if cost_reference == 0:
        cost_reference = 1 ##impedir divisao por 0 mais a frente

    resource_references = []
    for l in range(instance.m):
        if normalization == "node":
            total_resource = sum(resources[l] for resources in instance.node_resources.values()) ##soma os recursos totais de todos os nos e faz media
            reference = instance.k * total_resource / instance.n

        elif normalization == "edge":
            total_consumption = sum(edge.resources[l] for edge in instance.edges.values())
            reference = total_consumption / len(instance.edges)   ##forma nova soma os consumos totais de todos as edges e faz media
        
        if reference == 0:
            reference = 1  ##impedir divisao por 0 mais a frente
        resource_references.append(reference)


    ##percorrer os nos e contruir a efetivamente a solução
    while len(selected_nodes) < instance.k:
        candidates = []

        for edge in instance.edges.values():
            if edge.u in selected_nodes and edge.v not in selected_nodes:
                new_node = edge.v
            elif edge.v in selected_nodes and edge.u not in selected_nodes:
                new_node = edge.u
            else:
                continue

            new_slack = []
            for l in range(instance.m):
                remaining = slack[l]+instance.node_resources[new_node][l]-edge.resources[l]
                new_slack.append(remaining)

            #if any(value < 0 for value in new_slack):
            #    continue

            candidates.append((new_node,edge, new_slack))


        if not candidates:
            return None

        ##escolher o novo nó com o sistema de pontuação que eu decidi
        ##score=custo-folga (normalizei para poder compara melhor e o lambda é para mudar o peso das coisas)
        ##quanto menor melhor
        best_candidate = None
        best_key = None
        scored_candidates = []    ##para o random guarda os scores tb
        for new_node, edge, new_slack in candidates:
            normalized_cost = edge.cost / cost_reference
            minimum_slack = min (new_slack[l]/resource_references[l] for l in range(instance.m)) ## no caso de mais que um recurso escolher o minimo eventualmente mudar isto
            score = normalized_cost - lvalue * minimum_slack

            ##comparaçao primeiro por score, depois custo, depois numero do no, depois identificadores
            selection_key=(score, edge.cost, new_node, edge_key(edge.u, edge.v))    
            #if best_key == None or selection_key < best_key:
            #    best_key = selection_key
            #    best_candidate = (new_node,edge, new_slack)
            scored_candidates.append((selection_key, new_node, edge, new_slack))  ##guarda tudo 

        if alpha == None:
            best_entry = min(scored_candidates, key=lambda entry: entry[0])    ##contruçao normal
        else:
            scores = [entry[0][0] for entry in scored_candidates]
            minimum_score=min(scores)
            maximum_score=max(scores)
            threshold = minimum_score + alpha*(maximum_score-minimum_score)
            rcl=[entry for entry in scored_candidates if entry[0][0] <= threshold]  ##escolher os melhores considerando a condição dos slides e depois random uma delas
            best_entry=rng.choice(rcl)


        best_candidate = best_entry[1:]
        ##por candidato na tree
        new_node, edge, new_slack = best_candidate
        selected_nodes.add(new_node)
        selected_edges.append(edge_key(edge.u,edge.v))
        slack=new_slack
        total_cost += edge.cost

    return selected_edges

## percorrer o algoritmo de construçao começando em cada nó e escolher o de custo mais baixo e que seja valido
def construct_multistart(instance: Instance, lvalue: float = 8, normalization:str = "node"):
    best_solution = None
    best_cost = float("inf")

    for start in range(1, instance.n + 1):
        solution = construct_heu(instance, start, lvalue, normalization)

        if solution is None:
            continue

        result = validate_solution(instance, solution)

        if result.valid and result.cost < best_cost:
            best_solution = solution
            best_cost = result.cost

    return best_solution

def construct_randomized(instance: Instance, iterations: int = 100, alpha: float = 0.2, lvalue: float = 8, normalization: str = "node", seed: int = 42):
    rng = random.Random(seed)
    best_solution = None
    best_cost = float ("inf")
    valid_count = 0
    unique_solutions = set()

    for i in range(iterations):
        start = rng.randint(1, instance.n)
        solution =construct_heu(instance, start, lvalue=lvalue, normalization=normalization, alpha=alpha, rng=rng)
        if solution is None:
            continue

        result=validate_solution(instance, solution)

        if result.valid:
            valid_count+=1
            signature = frozenset(edge_key(u, v) for u, v in solution) ##guardar o numero de soluções validas e diferentes
            unique_solutions.add(signature) 
        
            if result.cost < best_cost:
                best_solution = solution
                best_cost = result.cost
        
    print("Tentativas:", iterations)
    print("Construções válidas:", valid_count)
    print("Árvores válidas diferentes:", len(unique_solutions))
    
    return best_solution


## construir a partir de uma árvore parcial.
## Faz até gamma novas adições, sem alterar a árvore inicial
## árvore devolvida pode tar incompleta ou com défices de recursos.
def pilot_rollout(instance: Instance, start:int, initial_edges: list[tuple[int, int]], lvalue:float=8, normalization:str = "node", gamma:int =3):    
    selected_edges= list(initial_edges)  ## copia para nao alterar a lista principal
    selected_nodes={start}

    for u,v in selected_edges: ##ir buscar os nos ja escolhidos (set evita repetir)
        selected_nodes.add(u)
        selected_nodes.add(v)  

    slack=[] ##folga atual
    for l in range(instance.m):
        available = sum(instance.node_resources[v][l] for v in selected_nodes)  ## ver os recursos ate agr
        consumed= sum(instance.edges[edge_key(u,v)].resources[l] for u,v in selected_edges) ##ver os recursos consumidos ate agr
        slack.append(available-consumed)

    steps=0
    ##igual ao alg de construção
    ####calculo normalizado para depois comparar na pontuação, ver se ha melhor alternativa
    cost_reference = (sum(edge.cost for edge in instance.edges.values()) / len(instance.edges)) ##media
    if cost_reference == 0:
        cost_reference = 1 ##impedir divisao por 0 mais a frente

    resource_references = []
    for l in range(instance.m):
        if normalization == "node":
            total_resource = sum(resources[l] for resources in instance.node_resources.values()) ##soma os recursos totais de todos os nos e faz media
            reference = instance.k * total_resource / instance.n

        elif normalization == "edge":
            total_consumption = sum(edge.resources[l] for edge in instance.edges.values())
            reference = total_consumption / len(instance.edges)   ##forma nova soma os consumos totais de todos as edges e faz media
        
        if reference == 0:
            reference = 1  ##impedir divisao por 0 mais a frente
        resource_references.append(reference)

    ## Parar ao atingir k nós ou após gamma novas adições
    ## As arestas recebidas em initial_edges não contam para gamma
    while len(selected_nodes) < instance.k and steps < gamma:
        best_candidate = None
        best_key = None

        #basicamente igual ao outro
        for edge in instance.edges.values():
            if edge.u in selected_nodes and edge.v not in selected_nodes:
                new_node = edge.v
            elif edge.v in selected_nodes and edge.u not in selected_nodes:
                new_node = edge.u
            else:
                continue

            ## Calcular as folgas que teríamos se adicionássemos este candidato
            new_slack = []
            for l in range(instance.m):
                remaining = slack[l]+instance.node_resources[new_node][l]-edge.resources[l]
                new_slack.append(remaining)

       
            normalized_cost = edge.cost / cost_reference
            minimum_slack = min (new_slack[l]/resource_references[l] for l in range(instance.m)) ## no caso de mais que um recurso escolher o minimo eventualmente mudar isto
            score = normalized_cost - lvalue * minimum_slack

            ##comparaçao primeiro por score, depois custo, depois numero do no, depois identificadores
            selection_key=(score, edge.cost, new_node, edge_key(edge.u, edge.v))    
            if best_key == None or selection_key < best_key:
                best_key = selection_key
                best_candidate = (new_node,edge, new_slack)
        #se nao houver ligaçao possivel
        if best_candidate is None:
            return None

        ## Aplicar a escolha apenas à cópia usada na simulação
        new_node, edge, new_slack = best_candidate
        selected_nodes.add(new_node)
        selected_edges.append(edge_key(edge.u, edge.v))
        slack = new_slack
        steps += 1

    return selected_edges

def construct_pilot(instance: Instance, start: int, gamma: int = 3, lvalue: float = 8, pilot_penalty: float = 8, normalization: str = "node"):
    selected_nodes={start}
    selected_edges=[]
    best_solution=None
    best_cost=float("inf")
    
    ##igual ao alg de construção neste caso para normalizar o custa das adiçoes simuladas
    cost_reference = (sum(edge.cost for edge in instance.edges.values()) / len(instance.edges)) ##media
    if cost_reference == 0:
        cost_reference = 1 ##impedir divisao por 0 mais a frente
    
    resource_references = []
    for l in range(instance.m):
        if normalization == "node":
            total_resource = sum(resources[l] for resources in instance.node_resources.values()) ##soma os recursos totais de todos os nos e faz media
            reference = instance.k * total_resource / instance.n
    
        elif normalization == "edge":
            total_consumption = sum(edge.resources[l] for edge in instance.edges.values())
            reference = total_consumption / len(instance.edges)   ##forma nova soma os consumos totais de todos as edges e faz media
            
        if reference == 0:
            reference = 1  ##impedir divisao por 0 mais a frente
        
        resource_references.append(reference)

    while len(selected_nodes) < instance.k:
        best_candidate = None
        best_key=None
        ##igual ao outro mas adicionar o pilot
        for edge in instance.edges.values():
            if edge.u in selected_nodes and edge.v not in selected_nodes:
                new_node = edge.v
            elif edge.v in selected_nodes and edge.u not in selected_nodes:
                new_node = edge.u
            else:
                continue
                
            ## criar a tree com o candidato
            trial_edges = selected_edges + [edge_key(edge.u,edge.v)]
            simulated_edges=pilot_rollout(instance,start=start,initial_edges=trial_edges,gamma=gamma,lvalue=lvalue,normalization=normalization)
            ###ir buscar os nos da arvore simulada
            simulated_nodes ={start}
            for u, v in simulated_edges:
                simulated_nodes.add(u)
                simulated_nodes.add(v)
            ##custo acrescentado desta simulação (candidato+rollout)
            added_edges = simulated_edges[len(selected_edges):] ##so as novas simuladas
            added_cost=sum(instance.edges[edge_key(u,v)].cost for u,v in added_edges)

            ##calcular defices mesmo q arvore nao completa
            total_deficit=0
            for l in range(instance.m):
                available=sum(instance.node_resources[v][l] for v in simulated_nodes)
                consumed=sum(instance.edges[edge_key(u,v)].resources[l] for u,v in simulated_edges)
                deficit= max(0,consumed-available)  ##caso nao haja defice nao se penaliza, é 0
                total_deficit+=deficit/resource_references[l]
                 
            evaluation=added_cost/cost_reference + pilot_penalty*total_deficit  ##forma de avaliar que escolhi

            priority=0
            if len(simulated_nodes) == instance.k:
                result=validate_solution(instance, simulated_edges)
                if result.valid:
                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_solution = list(simulated_edges)
                else:
                    priority=1

            ###priority é para o caso de nao encontrar soluções validas comparar na mesma as invalidas porque depois podem deixar de ser
            selection_key=(priority, evaluation, edge.cost, new_node, edge_key(edge.u, edge.v))

            ##nao percebo a necessidade disto

            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_candidate = (new_node, edge)

        if best_candidate is None:
            break

        ##adicionar o primeiro (pilot method)
        new_node, edge = best_candidate
        selected_nodes.add(new_node)
        selected_edges.append(edge_key(edge.u,edge.v))

    return best_solution

def pilot_multistart(instance: Instance, gamma: int | None = None, lvalue: float = 8, pilot_penalty: float = 8, normalization: str = "node"):
    ## Por defeito, permitir rollouts até completar a árvore
    if gamma is None:
        gamma = instance.k

    best_solution = None
    best_cost = float("inf")

    for start in range(1, instance.n + 1):
        solution = construct_pilot(instance, start=start, gamma=gamma, lvalue=lvalue, pilot_penalty=pilot_penalty, normalization=normalization)
        
        if solution is None:
            continue

        result = validate_solution(instance, solution)

        if result.valid and result.cost < best_cost:
            best_cost = result.cost
            best_solution = list(solution)

    return best_solution

##o que é esta porra
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    args = parser.parse_args()
    instance = read_instance(args.instance)
    print(f"Instance: {instance.name}")
    print(f"Nodes: {instance.n} | Edges: {len(instance.edges)}")
    print(f"Nodes needed: {instance.k} | Types of resources: {instance.m}")


if __name__ == "__main__":
    main()
