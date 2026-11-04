from time import perf_counter
from problem import edge_key, validate_solution
from neighborhoods import edge_exchange_neighbors, leaf_exchange_neighbors, internal_to_leaf_neighbors


def vnd(instance, initial_solution,neighborhoods=None, step_function="best",time_limit=30):
    begin=perf_counter()
    deadline=begin+time_limit
    ##ordem default mas possivel defini-la
    if neighborhoods is None:
        neighborhoods=[edge_exchange_neighbors,leaf_exchange_neighbors,internal_to_leaf_neighbors]

    validation=validate_solution(instance,initial_solution)
    if not validation.valid:
        raise ValueError("Initial solution invalid")

    ##sol inicial copiada para nao a alterar caso nao haja melhorias
    current_edges=sorted(edge_key(u,v) for u,v in initial_solution)
    current_cost=validation.cost
    current_slack=validation.slack
    initial_cost=current_cost #**

    ##contadores para estatisticas
    neighborhood_index=0
    iterations=0
    evaluated_neighbors=0
    stop_reason="time limit"
    stats=[{"name": neighborhood.__name__,"calls":0,"moves":0,"cost_reduction":0,"evaluated_valid_neighbors":0} for neighborhood in neighborhoods]

    while neighborhood_index < len(neighborhoods):
        if perf_counter() >= deadline:
            break

        neighborhood=neighborhoods[neighborhood_index]
        current_stats=stats[neighborhood_index]
        current_stats["calls"]+=1

        best_move=None
        best_move_cost=current_cost

        ##começar a procura na vizinhança index atual
        for move in neighborhood(instance,current_edges,current_cost,current_slack,deadline):
            if perf_counter()>=deadline:
                break

            removed_edges, added_edges, new_cost,new_slack = move
            evaluated_neighbors+=1
            current_stats["evaluated_valid_neighbors"]+=1

            if new_cost < best_move_cost:
                best_move=move
                best_move_cost=new_cost

                if step_function == "first":
                    break
                ## se for best continua tal como no local search
        
        timed_out=perf_counter()>=deadline

        if best_move is None:
            if timed_out:
                break  ##se o tempo acabar (time_out) para tudo
            ## se esta vizinhança nao melhorou bora de proxima
            neighborhood_index+=1
            continue
        ## se tiver havido melhoria atualizar 
        removed_edges,added_edges,new_cost,new_slack=best_move
        updated_edges=set(current_edges)
        for u,v in removed_edges:
            updated_edges.remove(edge_key(u,v))
        for u,v in added_edges:
            updated_edges.add(edge_key(u,v))

        ##atualizar estatisitica
        current_stats["moves"]+=1
        current_stats["cost_reduction"] += current_cost - new_cost
        current_edges=sorted(updated_edges)
        current_cost=new_cost
        current_slack=new_slack
        iterations+=1

        ##sempre que melhora o vnd volta ao inicio
        neighborhood_index=0
        if timed_out:
            break

    if neighborhood_index==len(neighborhoods):
        stop_reason="local optimum" ##se percorreu todas as viznhanças e nao consegue diminuir mais o custo

    final_validation = validate_solution(instance, current_edges)

    if not final_validation.valid:
        raise RuntimeError("VND invalid solution.")
    if (final_validation.cost != current_cost or final_validation.slack != current_slack):
        raise RuntimeError("Error on cost or slack")

    information = {"strategy": step_function, "initial_cost": initial_cost, "final_cost": current_cost, "iterations": iterations, "evaluated_valid_neighbors": evaluated_neighbors, "stop_reason": stop_reason, "neighborhood_stats": stats, "time": perf_counter() - begin}

    return current_edges, information
