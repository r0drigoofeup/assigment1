from time import perf_counter
from problem import Instance, edge_key


##Cria uma neighbhoroud retirar uma aresta e adicionar outra mas mantendo os mesmos nós
def edge_exchange_neighbors(instance: Instance, selected_edges, current_cost,current_slack,deadline):
    ##uniformizar os extremos das arestas
    tree_edges={edge_key(u,v) for u,v in selected_edges}
    selected_nodes = set()
    for u, v in tree_edges:
        selected_nodes.add(u)
        selected_nodes.add(v)

    ### save the neigh of each node on the current tree
    adjacency = {v: [] for v in selected_nodes} ## para cada no criar uma lista vazia e depois percorrer as edges e por no dicionario
    for u,v in sorted(tree_edges):
        adjacency[u].append(v)
        adjacency[v].append(u)

    for removed_key in sorted(tree_edges):
        if perf_counter()>= deadline:
            return

        ##ver os nos de cada lado considerandoq q removemos uma aresta [removed_key]
        removed_edge = instance.edges[removed_key]
        root = removed_key[0]
        component = {root}
        stack = [root]

        ## basicamente descobrir um dos lados da arvore ex: 1-2-3-4 se retirar a lig 2-3 fico com dois lados 1,2 e 3,4
        while stack:
            if perf_counter()>=deadline:
                return
            node=stack.pop() ##cada no que vamos explorando tiramos da stack


            for neighbor in adjacency[node]:
                if edge_key(node,neighbor) == removed_key: ##ignorar a ligacao que tiramos
                    continue
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)

        for added_key, added_edge in instance.edges.items(): ##dicionario vai buscar a chave+class em si
            u,v = added_key
            ###não pode ser a aresta ja escolhida e nao pode ser a proprira retirada (voltar a po la da a mesma solução) e os dois extremos tem de pertencer aos nos escolhidos e se estão do mesmo lado
            if added_key in tree_edges: #nao adicinar uma aresta que ja pertence/a retirada
                continue
            if u not in selected_nodes or v not in selected_nodes: ##nao mudamos os nos escolhidos
                continue
            if (u in component) == (v in component): ## um extremo ta de uma lado da arvore e o outro esta do outro para a arvore ligar
                continue

            ##calcular novo custo
            new_cost=current_cost-removed_edge.cost+added_edge.cost
            ##calcular nova folga (como os nos sao os mesmo, os recursos disponiveis sao iguais logo e so atualizar o consumo das arestas)
            new_slack= tuple(current_slack[l]+removed_edge.resources[l]-added_edge.resources[l] for l in range(instance.m))

            if any(value<0 for value in new_slack):
                continue  ##folga tem de ser postiva sempre para sol ser valida

            yield([removed_key],[added_key],new_cost,new_slack)


#####retirar uma folha da arvore (folha é tipo um extrema so tem uma ligaçao) desta vez vamos trocar os nos (no caso adicionar um nó que nao os iniciais), logo os recursos dos nos vao mudar desta vez
def leaf_exchange_neighbors(instance, selected_edges, current_cost, current_slack, deadline):
    ##igual ao outro, ir buscar os nos e arestas da arvore
    tree_edges={edge_key(u,v) for u,v in selected_edges}
    selected_nodes = set()
    for u, v in tree_edges:
        selected_nodes.add(u)
        selected_nodes.add(v)
    
    ### save the neigh of each node on the current tree
    adjacency = {v: [] for v in selected_nodes} ## para cada nó criar uma lista vazia e depois percorrer as edges e por no dicionario
    for u,v in sorted(tree_edges):
        adjacency[u].append(v)
        adjacency[v].append(u)


    for leaf in sorted(selected_nodes):
        if perf_counter() >= deadline:
            return

        if len(adjacency[leaf]) !=1:
            continue ##nao é uma folha se a lista do dic tiver mais que 1

        parent =adjacency [leaf][0]  ##neigb da folha
        removed_key=edge_key(leaf,parent)
        removed_edge=instance.edges[removed_key]

        remaining_nodes= selected_nodes-{leaf}    #nos que ficam

        ##procurar uma aresta que ligue um novo no exterior à árvore
        for added_key, added_edge in instance.edges.items():
            if perf_counter() >= deadline:
                return
            
            u,v=added_key

            if u in remaining_nodes and v not in selected_nodes:
                new_node=v
            elif v in remaining_nodes and u not in selected_nodes:
                new_node=u
            else:
                continue

            ##retirar custos e consumos e arranjar recursos
            new_cost=current_cost-removed_edge.cost+added_edge.cost
            new_slack=tuple(current_slack[l]-instance.node_resources[leaf][l]+instance.node_resources[new_node][l]+removed_edge.resources[l]-added_edge.resources[l] for l in range(instance.m))

            if any(value<0 for value in new_slack):
                continue  ##folga tem de ser postiva sempre para sol ser valida

            yield([removed_key],[added_key],new_cost,new_slack) ##ja entendi o yield


##aqui tiramos um no do meio (tem 2 ligações) e juntamos as duas ligaçoes abertas e depois acrecentamos outro nó (leaf que so tem uma ligação)
def internal_to_leaf_neighbors(instance,selected_edges,current_cost,current_slack,deadline):
    ##igual ao outro, ir buscar os nos e arestas da arvore
    tree_edges={edge_key(u,v) for u,v in selected_edges}
    selected_nodes = set()
    for u, v in tree_edges:
        selected_nodes.add(u)
        selected_nodes.add(v)
        
    ### save the neigh of each node on the current tree
    adjacency = {v: [] for v in selected_nodes} ## para cada nó criar uma lista vazia e depois percorrer as edges e por no dicionario
    for u,v in sorted(tree_edges):
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    
    for removed_node in sorted(selected_nodes):
        if perf_counter() >= deadline:
            return
        ##so retirar com 2 ligações
        if len(adjacency[removed_node])!=2:
            continue  

        a,b=adjacency[removed_node] ##os 2 vizinhos que queremos ligar caso haja essa ligaçao se nao bora de next
        bridge_key=edge_key(a,b)
        if bridge_key not in instance.edges:
            continue
        bridge_edge=instance.edges[bridge_key]
        ##pronto remover as arestas que desaparecem quando tiramos o nó
        removed_key_a=edge_key(a,removed_node)
        removed_key_b=edge_key(removed_node,b)
        removed_edge_a=instance.edges[removed_key_a]
        removed_edge_b=instance.edges[removed_key_b]
        remaining_nodes=selected_nodes-{removed_node}

        ##atualizar custos de retirar este nó
        base_cost=current_cost-removed_edge_a.cost-removed_edge_b.cost+bridge_edge.cost
        ## atualizar a folga (-do no retirado +consumos das edges removed - consumo da bridge)
        base_slack=tuple(current_slack[l]-instance.node_resources[removed_node][l]+removed_edge_a.resources[l]+removed_edge_b.resources[l]-bridge_edge.resources[l] for l in range(instance.m))

        ##procurar uma ligação entre nó exterior e a tree
        for added_key, added_edge in instance.edges.items():
            if perf_counter() >= deadline:
                    return
            
            u,v=added_key
            if u in remaining_nodes and v not in selected_nodes:
                new_node=v
            elif v in remaining_nodes and u not in selected_nodes:
                new_node=u
            else:
                continue

            ##atualizar custo e folga da nova edge
            new_cost=base_cost + added_edge.cost
            new_slack=tuple(base_slack[l]+instance.node_resources[new_node][l]-added_edge.resources[l] for l in range(instance.m))
    
            if any(value < 0 for value in new_slack):
                continue

            yield([removed_key_a, removed_key_b],[bridge_key,added_key],new_cost,new_slack)
