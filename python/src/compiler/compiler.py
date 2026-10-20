"""
The compiler: an ordered pipeline of passes, loaded once and run on source text.

A pass is a function `(program, system) -> program` tagged with `@compile_pass(takes, gives)`, where each is a level of
representation. The levels run in one direction, and a pass either stays on its level or moves to the next:

    NODE     the typed AST (a list of nodes)
    ISA      a flat list of instructions
    CIRCUIT  a quantum circuit

so a pipeline is node -> node passes, one lowering to the ISA, ISA -> ISA passes, one lowering to a circuit, then
circuit -> circuit passes. `load` checks that what each pass takes is what the one before it gives.

    compiler = Compiler(System(register_width=8, qubits=200)).load(inline, unroll, lower_to_isa, allocate, to_openqasm)
    result = compiler.run(source)      # result.stages has every pass's output, result.circuit the last one

Passes know nothing about each other, and a pass can be called directly like any function (`inline(nodes)`); the
system is optional for any pass that does not need one.
"""
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from compiler.parser.parser import parse
from compiler.system import System

NODE, ISA, CIRCUIT = "node", "isa", "circuit"
LEVELS = (NODE, ISA, CIRCUIT)


@dataclass(frozen=True)
class PassSpec:
    name: str  # the pass's id: unique within a compiler, the function's name by default
    takes: str
    gives: str
    title: str  # what a person reads: "Inline functions"
    summary: str = ""  # what the pass does, in a sentence or two
    requires: Tuple[str, ...] = ()  # passes without which it fails or gives wrong results on some valid program


def compile_pass(takes: str, gives: str, name: Optional[str] = None, title: Optional[str] = None, summary: str = "",
                 requires: Sequence[str] = ()) -> Callable:
    """
    Tag a function as a pass that takes one level and gives the same or the next. `title` and `summary` are how a
    viewer names and explains it, so a demo reads them from the compiler instead of keeping its own table.
    `requires` lists only what the pass cannot work without: it refuses the program, or gives a wrong one, when they
    are left out. A preferred order is not a requirement, and neither is a dependency on the level a pass takes,
    which the compiler already enforces.
    """
    if takes not in LEVELS or gives not in LEVELS or LEVELS.index(gives) - LEVELS.index(takes) not in (0, 1):
        raise ValueError(f"a pass cannot take {takes} and give {gives}")

    def tag(function: Callable) -> Callable:
        pass_name = name or function.__name__
        function.compile_pass = PassSpec(pass_name, takes, gives, title or pass_name.replace("_", " ").capitalize(),
                                         " ".join(summary.split()), tuple(requires))
        return function
    return tag


@dataclass(frozen=True)
class Stage:
    name: str  # "source" or the pass
    level: str
    output: Any
    warnings: Tuple[str, ...] = ()
    title: str = ""  # the pass's title


class PassError(Exception):
    """A pass failed. `stages` is everything produced before it, and `partial` is what it could still show, if anything."""

    def __init__(self, name: str, stages: Sequence[Stage], cause: BaseException, partial: Any = None):
        super().__init__(f"{name}: {type(cause).__name__}: {cause}")
        self.name = name
        self.stages = tuple(stages)
        self.cause = cause
        self.partial = partial


@dataclass(frozen=True)
class Compilation:
    system: System
    stages: Tuple[Stage, ...]

    def __getitem__(self, name: str) -> Any:
        """The output of a pass, by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage.output
        raise KeyError(name)

    @property
    def names(self) -> List[str]:
        return [stage.name for stage in self.stages]

    @property
    def output(self) -> Any:
        return self.stages[-1].output

    @property
    def circuit(self) -> Any:
        """The last stage's output, if the program was taken as far as a circuit."""
        if self.stages[-1].level != CIRCUIT:
            raise ValueError(f"the compilation stopped at {self.stages[-1].name}, which is not a circuit")
        return self.output


class Compiler:
    def __init__(self, system: Optional[System] = None, grammar: str = "v1") -> None:
        self.system = system or System()
        self.grammar = grammar  # which syntax `run` reads the source with (see compiler.parser.parser.GRAMMARS)
        self.passes: List[Callable] = []

    def load(self, *passes: Callable) -> "Compiler":
        """Append passes in the order they run. Each must take the level the previous one gave (the first takes NODE)."""
        level = self.passes[-1].compile_pass.gives if self.passes else NODE
        names = {p.compile_pass.name for p in self.passes}
        for function in passes:
            spec = getattr(function, "compile_pass", None)
            if spec is None:
                raise ValueError(f"{function!r} is not a pass: tag it with @compile_pass")
            if spec.takes != level:
                raise ValueError(f"{spec.name} takes {spec.takes} but comes after a pass that gives {level}")
            if spec.name in names:
                raise ValueError(f"{spec.name} is loaded twice")
            names.add(spec.name)
            self.passes.append(function)
            level = spec.gives
        return self

    @property
    def names(self) -> List[str]:
        return [p.compile_pass.name for p in self.passes]

    def run(self, source: str, until: Optional[str] = None, skip: Sequence[str] = ()) -> Compilation:
        """
        Parse `source` and apply the loaded passes in order, stopping after `until` if given. Passes named in `skip`
        are left out, which a following pass may not accept: that is reported as a failure of that pass. A failing
        pass raises `PassError` carrying the stages before it.
        """
        if until is not None and until not in self.names:
            raise ValueError(f"no pass named {until}")
        output: Any = parse(source, self.grammar)
        level = NODE
        stages = [Stage("source", NODE, output, title="Source")]
        for function in self.passes:
            spec = function.compile_pass
            if spec.name in skip:
                continue
            if spec.takes != level:
                raise PassError(spec.name, stages, TypeError(
                    f"expects a {spec.takes} program but the stage before it produced a {level} one: "
                    f"a pass in between was skipped"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    output = function(output, self.system)
                except Exception as error:
                    raise PassError(spec.name, stages, error, getattr(error, "partial", None)) from error
            level = spec.gives
            stages.append(Stage(spec.name, level, output, tuple(str(w.message) for w in caught), spec.title))
            if spec.name == until:
                break
        return Compilation(self.system, tuple(stages))
