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
    {
        "id": "temporary",
        "name": "A temporary that can be undone",
        "description": "t is computed from a and read once, by a comparison that is not a step on t, so it keeps its own "
                       "register. t is released by running its multiplication backwards; a stays, because the result u "
                       "is entangled with it and cannot be computed back from u.",
        "source": """a = 0..4;
t = a * 3;
u = t < 5;
return u;
""",
    },
    {
        "id": "fused",
        "name": "Steps combined, then an input computed back",
        "description": "t is read only by u and both multiply by a constant, so they combine into one multiplication "
                       "and t is never computed. 15 is odd, so a is then computed back from the result and released, "
                       "and only u stays held.",
        "source": """a = 0..4;
t = a * 3;
u = t * 5;
return u;
""",
    },
    {
        "id": "shared",
        "name": "Two independent distributions",
        "description": "p and y are never read, so the passes drop them. Switch off 'Update in place', 'Fold constants' and "
                       "'Drop unread values' to see the release pass undo y, reset k, and run p's preparation backwards, "
                       "while a stays entangled with the result.",
        "source": """a = 0..3;
p = 0..2;
k = 2;
x = a + k;
y = p + k;
return x;
""",
    },
    {
        "id": "branch-copy",
        "name": "A conditional that copies a distribution",
        "description": "b is 1 when a is 1 and 0 otherwise, so b is entangled with a. The release pass leaves programs with "
                       "controlled regions as they are, so nothing is released.",
        "source": """a = 0..2;
b = 0;
if a {
    b = 1;
}
return b;
""",
    },
    {
        "id": "branch-arms",
        "name": "Arithmetic in both arms",
        "description": "x starts as a constant, so folding turns each arm into a constant and the condition into a register "
                       "$t1 holding a < 2. $t1 is computed before the branch and read only as its control, so the release "
                       "pass undoes it afterwards and only a and x stay held.",
        "source": """a = 0..4;
x = 2;
if a < 2 {
    x += 3;
} else {
    x -= 1;
}
return x;
""",
    },
    {
        "id": "branch-nested",
        "name": "A conditional inside a conditional",
        "description": "x is 3 when both a and b are 1, 1 when only a is, and 0 otherwise. Two nested regions, so the writes "
                       "are controlled on two registers. Switch 'Flatten nested controls' off to compare with the "
                       "conjunction registers it adds.",
        "source": """a = 0..2;
b = 0..2;
x = 0;
if a {
    if b {
        x = 3;
    } else {
        x = 1;
    }
}
return x;
""",
    },
    {
        "id": "branch-temporary",
        "name": "A temporary computed inside a branch",
        "description": "t exists only to feed x and is computed under the condition. Undoing it needs the reverse of its "
                       "steps under the same control, which the release pass does not do yet.",
        "source": """a = 0..3;
x = 1;
if a {
    t = a * 3;
    x = t + 1;
}
return x;
""",
    },
]
