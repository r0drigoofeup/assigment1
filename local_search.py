from time import perf_counter
from problem import Instance, edge_key, validate_solution
from neighborhoods import edge_exchange_neighbors


def local_search(instance:Instance, initial_solution, neighborhood=edge_exchange_neighbors, step_function:str ="first", time_limit:float=30):

    begin = perf_counter()
    deadline = begin + time_limit

    if step_function not in ("first", "best"):
        raise ValueError("Strategy deve ser 'first' ou 'best'.")
    
    ##confirmar que arvore inicial valida so porque sim 
    validation=validate_solution(instance, initial_solution)
    if not validation.valid:
        raise ValueError("Inital solution invalid" + "; ".join(validation.errors))

    current_edges=sorted(edge_key(u,v) for u,v in initial_solution) ##copiar para nao alterar a sol input

    current_cost=validation.cost
    current_slack=validation.slack
    initial_cost=current_cost
    iterations=0  ##benchmarking
    evaluated_neighbors=0   ##benchmarking
    stop_reason ="time limit"

    while perf_counter() < deadline:
        best_move = None
        best_move_cost=current_cost

        for move in neighborhood(instance,current_edges,current_cost,current_slack,deadline):
            if perf_counter() >= deadline:
                break

            removed_edges,added_edges,new_cost,new_slack =move
            evaluated_neighbors+=1
            if new_cost<best_move_cost:
                best_move=move
                best_move_cost=new_cost

                if step_function=="first":
                    break  ##first improvement se nao for ou seja for best continua


        timed_out=perf_counter()>=deadline
        ## se o tempo acababar e nao ecnontar melhor solução é local opt
        if best_move is None:
            if timed_out:
                stop_reason ="time limit"
            else:
                stop_reason ="local optimum"
            break
        
        ##aplicar a melhoria escolhida
        removed_edges, added_edges, new_cost, new_slack=best_move
        updates_edges=set(current_edges)
        for u,v in removed_edges:
            updates_edges.remove(edge_key(u,v))
        for u,v in added_edges:
            updates_edges.add(edge_key(u,v))

        current_edges=sorted(updates_edges)
        current_cost=new_cost
        current_slack=new_slack
        iterations+=1

        if timed_out:
            stop_reason="time limit"
            break

    ##verificar que solução é valida
    final_validation=validate_solution(instance,current_edges)
    if not final_validation.valid:
        raise RuntimeError("Invalid solution from local search")

    if (final_validation.cost != current_cost or final_validation.slack != current_slack):
        raise RuntimeError("Error on updating resources/consumption")

    information = {"strategy": step_function, "initial_cost": initial_cost, "final_cost": current_cost, "iterations": iterations, "evaluated_valid_neighbors": evaluated_neighbors, "stop_reason": stop_reason, "time": perf_counter() - begin}


    return current_edges, information
