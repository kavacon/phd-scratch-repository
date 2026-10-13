"""The machine a program is compiled for. Every pass is given one."""
from dataclasses import dataclass


@dataclass(frozen=True)
class System:
    register_width: int = 6  # qubits in one register; integers are unsigned and arithmetic is modulo 2 to this
    qubits: int = 100  # qubits available in all

    def __post_init__(self) -> None:
        if self.register_width < 1:
            raise ValueError("register_width must be at least 1")
        if self.qubits < 0:
            raise ValueError("qubits cannot be negative")

    @property
    def slots(self) -> int:
        """How many registers a program may use: one register's worth of qubits is kept back for scratch."""
        return max(self.qubits // self.register_width - 1, 0)
