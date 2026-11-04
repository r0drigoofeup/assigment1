"""Testes pequenos para as regras do problema, antes de criar heurísticas."""

from pathlib import Path
import tempfile
import unittest

from rckst import read_instance, validate_solution
from rckst import Edge, Instance, pilot_rollout, construct_pilot, pilot_multistart


EXAMPLE = """6 8 4 1
#node resources
3
5
2
6
1
4
#edges
1 2 4 3
1 3 6 2
2 3 2 5
2 4 5 4
3 4 3 9
4 5 7 1
4 6 1 5
5 6 8 2
"""


class ProblemTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "example.txt"
        self.path.write_text(EXAMPLE, encoding="ascii")
        self.instance = read_instance(self.path)

    def test_statement_solution_and_reversed_endpoints(self):
        result = validate_solution(self.instance, [(3, 2), (2, 4), (6, 4)])
        self.assertTrue(result.valid, result.errors)
        self.assertEqual((result.cost, result.available, result.consumed, result.slack),
                         (8, (17,), (14,), (3,)))

    def test_cheaper_tree_exceeds_resource(self):
        result = validate_solution(self.instance, [(2, 3), (3, 4), (4, 6)])
        self.assertFalse(result.valid)
        self.assertEqual((result.cost, result.slack), (6, (-2,)))

    def test_duplicate_and_missing_edges(self):
        result = validate_solution(self.instance, [(1, 2), (2, 1), (1, 6)])
        self.assertFalse(result.valid)
        self.assertTrue(any("repeated" in error for error in result.errors))
        self.assertTrue(any("Non existent" in error for error in result.errors))

    def test_cycle_and_disconnection(self):
        result = validate_solution(self.instance, [(1, 2), (2, 3), (1, 3), (4, 6)])
        self.assertFalse(result.valid)
        self.assertTrue(any("cycles" in error for error in result.errors))
        self.assertTrue(any("dont form a connected structure" in error for error in result.errors))

    def test_resources_checked_separately(self):
        self.path.write_text("2 1 2 2\n#node resources\n10 0\n0 1\n#edges\n1 2 1 1 2\n",
                             encoding="ascii")
        result = validate_solution(read_instance(self.path), [(1, 2)])
        self.assertEqual(result.slack, (9, -1))
        self.assertFalse(result.valid)

    def test_pilot_without_edges(self):
        instance = Instance("empty", 2, 2, 1, {1: (10,), 2: (10,)}, {})
        self.assertIsNone(pilot_rollout(instance, start=1, initial_edges=[]))
        self.assertIsNone(construct_pilot(instance, start=1))
        self.assertIsNone(pilot_multistart(instance))

    def test_pilot_blocked_rollout(self):
        ## A componente do início tem só dois nós, mas precisamos de três
        instance = Instance(
            "disconnected", 3, 3, 1,
            {1: (10,), 2: (10,), 3: (10,)},
            {(1, 2): Edge(1, 2, 1, (1,))},
        )
        self.assertIsNone(pilot_rollout(instance, start=1, initial_edges=[(1, 2)]))
        self.assertIsNone(construct_pilot(instance, start=1))
        self.assertIsNone(pilot_multistart(instance))

    def test_pilot_still_returns_valid_tree(self):
        solution = pilot_multistart(self.instance)
        self.assertIsNotNone(solution)
        self.assertTrue(validate_solution(self.instance, solution).valid)

if __name__ == "__main__":
    unittest.main()
