"""
The programs this experiment is run on, in the same shape as the other experiments' (id, name, description, source).
The default register is 4 bits wide, so values run from -8 to 7 and results wrap. None of them has a release or
cleanup strategy behind it: temporaries are simply left in their registers.
"""
from typing import Dict, List

PROGRAMS: List[Dict[str, str]] = [
    {
        "id": "arithmetic",
        "name": "Arithmetic on a distribution",
        "description": "Everything is computed from one distribution, so every register ends up entangled with the others. "
                       "The results wrap at 4 bits.",
        "source": """a = 0..4;
b = a + 3;
c = b * 2;
return c;
""",
    },
    {
        "id": "mixed",
        "name": "A distribution beside definite values",
        "description": "One chain is computed from a distribution and one only from constants. The distribution's chain "
                       "is entangled; the constants' chain is not, even though it feeds the result.",
        "source": """a = 0..3;
b = a + 1;
k = 1;
m = k * 2;
c = m + 1;
d = b + c;
return d;
""",
    },
]
