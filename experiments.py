"""Comparar ordens, estratégias e número de vizinhanças do VND."""

import argparse
import csv
import json
import random
from itertools import product
from datetime import datetime
from pathlib import Path
from time import perf_counter

from rckst import (
    read_instance,
    construct_heu,
    construct_multistart,
    construct_randomized,
    validate_solution,
    vnd,
    edge_exchange_neighbors,
    leaf_exchange_neighbors,
    internal_to_leaf_neighbors,
)


ROOT = Path(__file__).resolve().parent


def default_cases():
    ## Repetir as soluções iniciais que já usámos nos testes manuais
    ## None significa multistart; nos casos grandes usamos um único início
    return [
        (ROOT / "instances/instances/train/100/n100_k30_m2_s10002.txt", None),
        (ROOT / "instances/instances/train/200/n200_k60_m3_s10004.txt", None),
        (ROOT / "instances/instances/train/500/n500_k150_m2_s10007.txt", 1),
        (ROOT / "instances/instances/train/1000/n1000_k300_m3_s10005.txt", 7),
    ]


def configurations():
    ## 2 ordens x 2 estratégias x 2 quantidades = 8 execuções por instância
    orders = [
        ("arestas_primeiro", [edge_exchange_neighbors, leaf_exchange_neighbors]),
        ("folhas_primeiro", [leaf_exchange_neighbors, edge_exchange_neighbors]),
    ]
    for order_name, pair in orders:
        for count in (2, 3):
            neighborhoods = list(pair)
            if count == 3:
                neighborhoods.append(internal_to_leaf_neighbors)
            for strategy in ("first", "best"):
                yield order_name, neighborhoods, strategy


FIELDS = [
    "instance", "path", "n", "k", "m", "construction", "start",
    "lambda", "normalization", "construction_time_s", "initial_cost",
    "order", "neighborhood_count", "strategy", "time_limit_s", "status",
    "valid", "final_cost", "improvement_percent", "moves",
    "evaluated_valid_neighbors", "stop_reason", "vnd_time_s", "slack",
    "neighborhood_stats", "errors",
]


def run_case(path, start, args, writer, output):
    instance = read_instance(path)
    limit = args.time_limit
    if limit is None:
        limit = 120 if instance.n >= 500 else 30

    print(f"\nInstância: {instance.name} | n={instance.n} | k={instance.k}", flush=True)
    print("A preparar a solução inicial...", flush=True)
    begin = perf_counter()
    if start is None:
        initial = construct_multistart(
            instance, lvalue=args.lvalue, normalization=args.normalization
        )
    else:
        if start not in instance.node_resources:
            raise ValueError(f"Nó inicial {start} inexistente em {instance.name}.")
        initial = construct_heu(
            instance, start=start, lvalue=args.lvalue,
            normalization=args.normalization,
        )
    construction_time = perf_counter() - begin

    base = {
        "instance": instance.name, "path": str(path.resolve()),
        "n": instance.n, "k": instance.k, "m": instance.m,
        "construction": "multistart" if start is None else "single_start",
        "start": start, "lambda": args.lvalue,
        "normalization": args.normalization,
        "construction_time_s": construction_time,
        "time_limit_s": limit,
    }

    ## Uma falha da construção também é um resultado: guardar, não esconder
    validation = None if initial is None else validate_solution(instance, initial)
    if validation is None or not validation.valid:
        writer.writerow({
            **base,
            "status": "no_initial_solution" if initial is None else "invalid_initial_solution",
            "valid": False,
            "initial_cost": "" if validation is None else validation.cost,
            "errors": "" if validation is None else json.dumps(validation.errors),
        })
        output.flush()
        print("Sem solução inicial válida; registado no CSV.", flush=True)
        return

    base["initial_cost"] = validation.cost
    print(f"Custo inicial: {validation.cost} | Limite por VND: {limit} s", flush=True)

    for order_name, neighborhoods, strategy in configurations():
        print(
            f"A executar: {order_name} | {strategy} | {len(neighborhoods)} vizinhanças",
            flush=True,
        )
        ## Todas as configurações recebem uma cópia da MESMA solução inicial
        solution, info = vnd(
            instance, list(initial), neighborhoods=neighborhoods,
            step_function=strategy, time_limit=limit,
        )
        result = validate_solution(instance, solution)
        improvement = (
            100 * (validation.cost - result.cost) / validation.cost
            if validation.cost != 0 else ""
        )
        writer.writerow({
            **base, "order": order_name,
            "neighborhood_count": len(neighborhoods), "strategy": strategy,
            "status": "ok" if result.valid else "invalid_vnd_solution",
            "valid": result.valid, "final_cost": result.cost,
            "improvement_percent": improvement, "moves": info["iterations"],
            "evaluated_valid_neighbors": info["evaluated_valid_neighbors"],
            "stop_reason": info["stop_reason"], "vnd_time_s": info["time"],
            "slack": json.dumps(result.slack),
            "neighborhood_stats": json.dumps(info["neighborhood_stats"]),
            "errors": json.dumps(result.errors),
        })
        ## Guardar após cada execução, mesmo que interrompamos mais tarde
        output.flush()
        print(
            f"Válida: {result.valid} | Custo: {validation.cost} → {result.cost}"
            f" | Trocas: {info['iterations']} | Paragem: {info['stop_reason']}"
            f" | Tempo: {info['time']:.4f} s",
            flush=True,
        )
        for stats in info["neighborhood_stats"]:
            print(
                f"  {stats['name']} | Trocas: {stats['moves']}"
                f" | Redução: {stats['cost_reduction']}",
                flush=True,
            )


def retry_failed(args, parser):
    ## Selecionar apenas falhas registadas, sem voltar a executar a construção original
    with args.retry_failed.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        if not {"status", "path"}.issubset(reader.fieldnames or []):
            parser.error("O CSV deve conter as colunas status e path.")
        rows = list(reader)
        ## Se qualquer seed já conseguiu solução, não repetir essa instância
        recovered = {Path(row["path"]) for row in rows if row["status"] == "ok"}
        paths = sorted({
            Path(row["path"])
            for row in rows
            if row["status"] in (
                "no_initial_solution", "invalid_initial_solution", "no_solution", "invalid_solution"
            )
        } - recovered)
    if not paths:
        print("O CSV não contém falhas da construção inicial.")
        return
    for path in paths:
        if not path.is_file():
            parser.error(f"Instância não encontrada: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = args.output or ROOT / "results" / f"recovery_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance", "path", "source_csv", "n", "k", "m", "seed",
        "iterations", "alpha", "lambda", "normalization", "status",
        "valid", "cost", "slack", "time_s", "solution_edges", "errors",
    ]
    with output_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=";")
        writer.writeheader()
        output.flush()
        print(f"Resultados: {output_path.resolve()}", flush=True)
        lvalues = args.lvalues if args.lvalues is not None else [args.lvalue]
        print(f"Instâncias com falhas: {len(paths)} | Seeds: 1, 2, 3 | Lambdas: {lvalues}", flush=True)
        try:
            for index, path in enumerate(paths, start=1):
                instance = read_instance(path)
                print(f"\nCaso {index}/{len(paths)}: {instance.name}", flush=True)
                ## Executar todas as seeds, mesmo que a primeira já encontre solução
                for lvalue, seed in product(lvalues, (1, 2, 3)):
                    begin = perf_counter()
                    solution = construct_randomized(
                        instance, iterations=100, alpha=0.05,
                        lvalue=lvalue, normalization=args.normalization,
                        seed=seed,
                    )
                    elapsed = perf_counter() - begin
                    result = None if solution is None else validate_solution(instance, solution)
                    valid = result is not None and result.valid
                    status = "ok" if valid else (
                        "no_solution" if result is None else "invalid_solution"
                    )
                    writer.writerow({
                        "instance": instance.name, "path": str(path.resolve()),
                        "source_csv": str(args.retry_failed.resolve()),
                        "n": instance.n, "k": instance.k, "m": instance.m,
                        "seed": seed, "iterations": 100, "alpha": 0.05,
                        "lambda": lvalue, "normalization": args.normalization,
                        "status": status, "valid": valid,
                        "cost": "" if result is None else result.cost,
                        "slack": "" if result is None else json.dumps(result.slack),
                        "time_s": elapsed,
                        ## Guardar a árvore permite aplicar o VND mais tarde
                        "solution_edges": json.dumps(solution),
                        "errors": "" if result is None else json.dumps(result.errors),
                    })
                    output.flush()
                    print(
                        f"Lambda: {lvalue} | Seed: {seed} | Estado: {status}"
                        f" | Custo: {result.cost if valid else '-'}"
                        f" | Tempo: {elapsed:.3f} s", flush=True,
                    )
        except KeyboardInterrupt:
            print("\nInterrompido. Os resultados concluídos ficaram guardados.")


def improve_recovered(args, parser):
    records = []
    ## Ler apenas árvores válidas; manter cada execução e a sua origem
    for source_path in dict.fromkeys(args.improve_recovered):
        if not source_path.is_file():
            parser.error(f"CSV não encontrado: {source_path}")
        with source_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            required = {"status", "path", "solution_edges", "seed", "lambda", "cost"}
            if not required.issubset(reader.fieldnames or []):
                parser.error(f"Não é um CSV de recuperação: {source_path}")
            records.extend(
                (source_path, row) for row in reader if row["status"] == "ok"
            )
    if not records:
        print("Não existem soluções recuperadas para melhorar.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = args.output or ROOT / "results" / f"recovered_vnd_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = FIELDS + ["source_csv", "seed", "alpha", "iterations", "solution_edges"]
    order = [leaf_exchange_neighbors, edge_exchange_neighbors, internal_to_leaf_neighbors]
    with output_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=";")
        writer.writeheader()
        output.flush()
        print(f"Resultados: {output_path.resolve()} | Execuções: {len(records)}", flush=True)
        try:
            for index, (source_path, row) in enumerate(records, start=1):
                instance = read_instance(row["path"])
                initial = json.loads(row["solution_edges"])
                validation = validate_solution(instance, initial)
                if not validation.valid or validation.cost != int(row["cost"]):
                    raise ValueError(f"Árvore guardada inválida ou custo incoerente: {instance.name}")
                limit = args.time_limit
                if limit is None:
                    limit = 120 if instance.n >= 500 else 30
                print(
                    f"\n{index}/{len(records)}: {instance.name}"
                    f" | Lambda: {row['lambda']} | Seed: {row['seed']}", flush=True,
                )
                solution, info = vnd(
                    instance, initial, neighborhoods=order,
                    step_function="best", time_limit=limit,
                )
                result = validate_solution(instance, solution)
                writer.writerow({
                    "instance": instance.name, "path": row["path"],
                    "n": instance.n, "k": instance.k, "m": instance.m,
                    "construction": "saved_randomized", "source_csv": str(source_path.resolve()),
                    "seed": row["seed"], "alpha": row.get("alpha", ""),
                    "iterations": row.get("iterations", ""), "lambda": row["lambda"],
                    "normalization": row.get("normalization", ""),
                    ## Tempo histórico da construção; não foi executada novamente
                    "construction_time_s": row.get("time_s", ""),
                    "initial_cost": validation.cost, "order": "folhas_primeiro",
                    "neighborhood_count": 3, "strategy": "best", "time_limit_s": limit,
                    "status": "ok" if result.valid else "invalid_vnd_solution",
                    "valid": result.valid, "final_cost": result.cost,
                    "improvement_percent": (
                        100 * (validation.cost - result.cost) / validation.cost
                        if validation.cost else ""
                    ),
                    "moves": info["iterations"],
                    "evaluated_valid_neighbors": info["evaluated_valid_neighbors"],
                    "stop_reason": info["stop_reason"], "vnd_time_s": info["time"],
                    "slack": json.dumps(result.slack),
                    "neighborhood_stats": json.dumps(info["neighborhood_stats"]),
                    "solution_edges": json.dumps(solution), "errors": json.dumps(result.errors),
                })
                output.flush()
                print(
                    f"Válida: {result.valid} | Custo: {validation.cost} → {result.cost}"
                    f" | Paragem: {info['stop_reason']} | Tempo: {info['time']:.4f} s",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\nInterrompido. Os resultados concluídos ficaram guardados.")


def compare_budgets(args, parser):
    ## Os dois orçamentos partilham as primeiras 50 tentativas por lambda
    ## Continuar até 100 evita repetir a primeira metade da experiência
    split = "test" if args.evaluate_test else "train"
    budgets = (50,) if args.evaluate_test else (50, 100)
    paths = args.instances or [
        path for size in (100, 200)
        for path in sorted((ROOT / f"instances/instances/{split}/{size}").glob("*.txt"))
    ]
    if not paths or any(not path.is_file() for path in paths):
        parser.error("Não foram encontradas todas as instâncias pedidas.")
    fields = [
        "instance", "path", "n", "k", "m", "seed", "attempts_per_lambda",
        "lambdas", "alpha", "normalization", "attempts_completed",
        "valid_constructions", "unique_valid_constructions", "vnd_calls",
        "vnd_timeouts", "best_construction_cost", "final_cost", "valid",
        "status", "stop_reason", "construction_time_s", "vnd_time_s",
        "total_time_s", "total_time_limit_s", "vnd_time_limit_s",
        "slack", "solution_edges",
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "test_fixed" if args.evaluate_test else "budgets"
    output_path = args.output or ROOT / "results" / f"{prefix}_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = [leaf_exchange_neighbors, edge_exchange_neighbors, internal_to_leaf_neighbors]
    lambdas = (8, 16, 32)
    with output_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=";")
        writer.writeheader()
        output.flush()
        print(f"Resultados: {output_path.resolve()} | Instâncias: {len(paths)}", flush=True)
        try:
            for index, path in enumerate(paths, start=1):
                instance = read_instance(path)
                per_vnd_limit = args.time_limit or (120 if instance.n >= 500 else 30)
                for seed in (1, 2, 3):
                    print(f"\n{index}/{len(paths)}: {instance.name} | Seed: {seed}", flush=True)
                    ## Uma sequência independente para cada lambda, reprodutível pela seed
                    generators = {l: random.Random(f"{seed}:{l}") for l in lambdas}
                    seen = set()
                    best_solution = None
                    best_cost = float("inf")
                    best_construction = float("inf")
                    attempts = valid_count = vnd_calls = vnd_timeouts = 0
                    construction_time = vnd_time = 0.0
                    begin = perf_counter()
                    ## Limite global de 15 minutos por instância/seed
                    deadline = begin + 900
                    previous_budget = 0
                    timed_out = False
                    for budget in budgets:
                        for attempt in range(previous_budget, budget):
                            if timed_out:
                                break
                            for lvalue in lambdas:
                                if perf_counter() >= deadline:
                                    timed_out = True
                                    break
                                rng = generators[lvalue]
                                start = rng.randint(1, instance.n)
                                construction_begin = perf_counter()
                                initial = construct_heu(
                                    instance, start=start, lvalue=lvalue,
                                    normalization="node", alpha=0.05, rng=rng,
                                )
                                validation = None if initial is None else validate_solution(instance, initial)
                                construction_time += perf_counter() - construction_begin
                                attempts += 1
                                if validation is None or not validation.valid:
                                    continue
                                valid_count += 1
                                best_construction = min(best_construction, validation.cost)
                                ## A própria construção é uma solução disponível se acabar o tempo
                                if validation.cost < best_cost:
                                    best_solution = list(initial)
                                    best_cost = validation.cost
                                signature = frozenset(tuple(sorted(edge)) for edge in initial)
                                if signature in seen:
                                    continue
                                seen.add(signature)
                                remaining = deadline - perf_counter()
                                if remaining <= 0:
                                    timed_out = True
                                    break
                                vnd_begin = perf_counter()
                                solution, info = vnd(
                                    instance, initial, neighborhoods=order, step_function="best",
                                    time_limit=min(per_vnd_limit, remaining),
                                )
                                vnd_time += perf_counter() - vnd_begin
                                vnd_calls += 1
                                vnd_timeouts += info["stop_reason"] == "time limit"
                                if info["final_cost"] < best_cost:
                                    best_solution = list(solution)
                                    best_cost = info["final_cost"]
                            if (attempt + 1) % 10 == 0:
                                print(f"  Tentativas por lambda: {attempt + 1} | VND executados: {vnd_calls}", flush=True)
                        total_time = perf_counter() - begin
                        result = None if best_solution is None else validate_solution(instance, best_solution)
                        if result is not None and (not result.valid or result.cost != best_cost):
                            raise RuntimeError("Solução final inválida ou custo incoerente.")
                        writer.writerow({
                            "instance": instance.name, "path": str(path.resolve()),
                            "n": instance.n, "k": instance.k, "m": instance.m, "seed": seed,
                            "attempts_per_lambda": budget, "lambdas": json.dumps(lambdas),
                            "alpha": 0.05, "normalization": "node", "attempts_completed": attempts,
                            "valid_constructions": valid_count, "unique_valid_constructions": len(seen),
                            "vnd_calls": vnd_calls, "vnd_timeouts": vnd_timeouts,
                            "best_construction_cost": "" if best_construction == float("inf") else best_construction,
                            "final_cost": "" if result is None else result.cost,
                            "valid": result is not None, "status": "ok" if result is not None else "no_solution",
                            "stop_reason": "time limit" if timed_out else "attempt budget",
                            "construction_time_s": construction_time, "vnd_time_s": vnd_time,
                            "total_time_s": total_time, "total_time_limit_s": 900,
                            "vnd_time_limit_s": per_vnd_limit,
                            "slack": "" if result is None else json.dumps(result.slack),
                            "solution_edges": json.dumps(best_solution),
                        })
                        output.flush()
                        print(
                            f"Orçamento: {budget} | Custo: {result.cost if result else '-'}"
                            f" | Tempo total acumulado: {total_time:.3f} s", flush=True,
                        )
                        if timed_out:
                            break
                        previous_budget = budget
        except KeyboardInterrupt:
            print("\nInterrompido. As medições já concluídas ficaram guardadas.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instances", nargs="*", type=Path,
        help="Ficheiros a testar; sem argumentos usa os quatro casos conhecidos.",
    )
    starts = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--compare-budgets", action="store_true",
        help="Comparar 50/100 tentativas por lambda (8,16,32), seeds 1/2/3 e VND em cada árvore válida diferente. Por defeito usa treino de 100/200 nós.",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Avaliar a configuração fixa (50 tentativas por lambda 8/16/32, alpha 0.05, três seeds e VND best) no teste de 100/200 nós.",
    )
    parser.add_argument(
        "--improve-recovered", type=Path, nargs="+",
        help="Aplicar VND (folhas primeiro, best, três vizinhanças) às árvores guardadas nos CSV de recuperação.",
    )
    parser.add_argument(
        "--retry-failed", type=Path,
        help="CSV anterior: testar construção aleatorizada nas instâncias sem solução inicial, sem executar VND.",
    )
    parser.add_argument(
        "--small-train", action="store_true",
        help="Testar todo o treino de 100 e 200 nós, com construção multistart por defeito.",
    )
    starts.add_argument("--start", type=int, help="Usar este nó inicial em todos os casos.")
    starts.add_argument("--multistart", action="store_true", help="Testar todos os nós iniciais.")
    parser.add_argument("--lvalue", type=float, default=8)
    parser.add_argument(
        "--lvalues", type=float, nargs="+",
        help="Valores de lambda a comparar no modo --retry-failed.",
    )
    parser.add_argument("--normalization", choices=("node", "edge"), default="node")
    parser.add_argument(
        "--time-limit", type=float,
        help="Limite por VND; por defeito 30 s para n<500 e 120 s para n>=500. Não limita a construção.",
    )
    parser.add_argument("--output", type=Path, help="CSV de saída (tem de ser um ficheiro novo).")
    args = parser.parse_args()
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit tem de ser positivo.")
    if args.evaluate_test and (args.compare_budgets or args.instances):
        parser.error("--evaluate-test usa apenas as instâncias de teste predefinidas.")
    if args.compare_budgets or args.evaluate_test:
        if args.retry_failed or args.improve_recovered or args.small_train or args.start is not None or args.multistart or args.lvalues is not None or args.lvalue != 8 or args.normalization != "node":
            parser.error("--compare-budgets usa parâmetros de construção fixos e não se combina com outros modos.")
        compare_budgets(args, parser)
        return
    if args.improve_recovered is not None:
        if args.retry_failed or args.instances or args.small_train or args.start is not None or args.multistart or args.lvalues is not None:
            parser.error("--improve-recovered não se combina com outros modos de seleção ou construção.")
        improve_recovered(args, parser)
        return
    if args.lvalues is not None and args.retry_failed is None:
        parser.error("--lvalues só pode ser usado com --retry-failed.")
    if args.retry_failed is not None:
        if args.instances or args.small_train or args.start is not None or args.multistart or args.time_limit is not None:
            parser.error("--retry-failed não se combina com seleção de instâncias, início ou limite do VND.")
        if not args.retry_failed.is_file():
            parser.error(f"CSV não encontrado: {args.retry_failed}")
        retry_failed(args, parser)
        return
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit tem de ser positivo.")

    ## Em ficheiros escolhidos pelo utilizador, começar no nó 1 por defeito
    if args.small_train:
        if args.instances:
            parser.error("Usa --small-train ou ficheiros explícitos, não ambos.")
        cases = []
        for size in (100, 200):
            paths = sorted((ROOT / f"instances/instances/train/{size}").glob("*.txt"))
            if not paths:
                parser.error(f"Não foram encontradas instâncias de treino de {size} nós.")
            cases.extend((path, None) for path in paths)
    else:
        cases = [(path, 1) for path in args.instances] if args.instances else default_cases()
    if args.multistart:
        cases = [(path, None) for path, _ in cases]
    elif args.start is not None:
        cases = [(path, args.start) for path, _ in cases]
    for path, _ in cases:
        if not path.is_file():
            parser.error(f"Ficheiro não encontrado: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = args.output or ROOT / "results" / f"vnd_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ## Modo x impede substituir resultados de uma experiência anterior
    with output_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter=";")
        writer.writeheader()
        output.flush()
        print(f"Resultados: {output_path.resolve()}", flush=True)
        print(f"Instâncias: {len(cases)} | Configurações por solução inicial válida: 8", flush=True)
        try:
            for index, (path, start) in enumerate(cases, start=1):
                print(f"\nCaso {index}/{len(cases)}", flush=True)
                run_case(path, start, args, writer, output)
        except KeyboardInterrupt:
            print("\nInterrompido. Os resultados já concluídos ficaram guardados.")


if __name__ == "__main__":
    main()
