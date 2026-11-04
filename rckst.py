

import argparse
from pathlib import Path

from problem import Edge, Instance, Validation, edge_key, read_instance, validate_solution
from construction import construct_heu, construct_multistart, construct_randomized
from pilot import pilot_rollout, construct_pilot, pilot_multistart
from neighborhoods import edge_exchange_neighbors, leaf_exchange_neighbors, internal_to_leaf_neighbors
from local_search import local_search
from vnd import vnd



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
