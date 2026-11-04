import random
from problem import Instance, edge_key, validate_solution


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
