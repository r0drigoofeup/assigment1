from problem import Instance, edge_key, validate_solution


## construir a partir de uma árvore parcial.
## Faz até gamma novas adições, sem alterar a árvore inicial
## árvore devolvida pode tar incompleta ou com défices de recursos.
def pilot_rollout(instance: Instance, start:int, initial_edges: list[tuple[int, int]], lvalue:float=8, normalization:str = "node", gamma:int =3):    
    ## Sem arestas não conseguimos construir a árvore nem calcular as médias
    if not instance.edges:
        return None

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
    ## Evitar divisão por zero quando o grafo não tem arestas
    if not instance.edges:
        return None

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
            ## Se a simulação ficou bloqueada experimentar outro candidato
            if simulated_edges is None:
                continue

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

        ##adicionar o primeiro (pila method)
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
