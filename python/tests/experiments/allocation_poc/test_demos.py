import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from socketserver import ThreadingMixIn

from checking import qiskit_tools
from experiments.allocation_poc import experiment
from experiments.allocation_poc.demos import compare, explorer
from experiments.allocation_poc.programs import PROGRAMS
from demo import app
from demo.model import Action, to_data, walk

KNOWN = {"code", "diff", "metrics", "stats", "switcher", "grid", "timeline", "note", "action", "diagram", "columns"}
needs_qiskit = unittest.skipUnless(qiskit_tools.qiskit_available(), "Qiskit is not installed")
needs_aer = unittest.skipUnless(qiskit_tools.qiskit_available() and qiskit_tools.aer_available(), "Qiskit Aer is not installed")


def blocks_of(result):
    return list(walk([block for stage in result.stages for block in stage.blocks]))


class ExplorerDemoTestCase(unittest.TestCase):
    """The demo only describes what the experiment returned, so these check the description is well formed."""

    def test_every_program_describes_to_known_blocks_that_serialise(self):
        for case in explorer.DEMO.cases:
            with self.subTest(program=case.id):
                result = explorer.run(explorer.DEMO.clean_inputs({"case": case.id}), [s.id for s in explorer.DEMO.steps])
                self.assertIsNone(result.error)
                self.assertEqual("source", result.stages[0].id)
                for block in blocks_of(result):
                    self.assertIn(to_data(block)["type"], KNOWN)
                json.dumps(to_data(result))

    def test_the_steps_are_the_passes_named_by_the_passes(self):
        passes = [fn.compile_pass for fn in experiment.compiler().passes]
        self.assertEqual([(p.name, p.title, p.summary, p.requires) for p in passes],
                         [(s.id, s.name, s.summary, s.requires) for s in explorer.DEMO.steps])
        self.assertTrue(all(p.summary for p in passes))

    def test_a_syntax_error_is_reported_against_the_input(self):
        inputs = explorer.DEMO.clean_inputs({"source": "function f(): int { return 1 + ; } return f();"})
        error = explorer.run(inputs, ["inline"]).error
        self.assertEqual((1, "Unexpected ';'."), (error.line, error.message))

    @needs_qiskit
    def test_circuit_stages_carry_counts_a_drawing_and_a_simulator_action(self):
        result = explorer.run(explorer.DEMO.clean_inputs({"case": "area"}), [s.id for s in explorer.DEMO.steps])
        for stage in (s for s in result.stages if s.id in ("to_openqasm", "optimise")):
            kinds = {to_data(b)["type"] for b in walk(stage.blocks)}
            self.assertTrue({"switcher", "diagram", "metrics", "action"} <= kinds, stage.id)
            action = next(b for b in walk(stage.blocks) if isinstance(b, Action))
            self.assertIn(action.action, explorer.DEMO.actions)

    @needs_aer
    def test_the_simulate_action_reports_against_the_expected_value(self):
        inputs = explorer.DEMO.clean_inputs({"case": "area"})
        blocks = explorer.simulate_action(inputs, [s.id for s in explorer.DEMO.steps], {"stage": "optimise", "shots": 20})
        self.assertIn("Every shot returned 13", blocks[0].text)


class CompareDemoTestCase(unittest.TestCase):

    @unittest.skipUnless(qiskit_tools.qiskit_available() and __import__("importlib").util.find_spec("qrisp"), "Qrisp and Qiskit are not installed")
    def test_it_describes_both_programs_the_statistics_and_the_circuits(self):
        result = compare.run(compare.DEMO.clean_inputs({"case": "expression", "width": 4}), [])
        self.assertEqual(["programs", "statistics", "circuits"], [s.id for s in result.stages])
        for block in blocks_of(result):
            self.assertIn(to_data(block)["type"], KNOWN)
        self.assertIn("stats", {to_data(b)["type"] for b in blocks_of(result)})


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(explorer.DEMO))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.url + path) as response:
            return response.read()

    def post(self, path, body):
        request = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def test_serves_the_shared_page_its_script_and_the_bootstrap(self):
        self.assertIn(b"app.js", self.get("/"))
        self.assertIn(b"BLOCKS", self.get("/app.js"))
        self.assertIn(b"circuitDiagram", self.get("/circuit.js"))
        boot = json.loads(self.get("/api/bootstrap"))
        self.assertEqual(len(PROGRAMS), len(boot["cases"]))
        self.assertEqual([s.id for s in explorer.DEMO.steps], [s["id"] for s in boot["steps"]])
        self.assertEqual("/api/action/simulate", boot["urls"]["actions"]["simulate"])


class SingleThreadTestCase(unittest.TestCase):
    def test_the_demo_servers_handle_requests_on_one_thread(self):
        # Qiskit's native allocator crashes the process when a new thread uses it after an earlier thread has finished,
        # which is what a thread per request does, so every request must run on the main thread.
        for demo in (explorer, compare):
            server = demo.make_server(0)
            try:
                self.assertNotIsInstance(server, ThreadingMixIn)
            finally:
                server.server_close()


if __name__ == '__main__':
    unittest.main()
