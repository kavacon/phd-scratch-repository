"""
The signed programs every encoding is run on, in the same shape as the allocation experiment's (id, name, description,
source). The default register is 4 bits wide, so values run from -8 to 7 and results wrap. Each program has no
release or cleanup strategy behind it: temporaries are simply left in their registers.

They cover each executable operation on its own, in-place updates, conditionals (including nested), loops, calls, and
whole programs that mix them.
"""
from typing import Dict, List

PROGRAMS: List[Dict[str, str]] = [
    {
        "id": "negatives",
        "name": "Add and subtract negatives",
        "description": "Signed add and subtract: the same circuits as unsigned, read as two's complement.",
        "source": """a = -3;
b = 5;
c = a + b;
d = c - 7;
return d;
""",
    },
    {
        "id": "multiply",
        "name": "Multiply with a negative",
        "description": "Multiplication is the same gates as unsigned; the product is read as two's complement.",
        "source": """a = 3;
b = -2;
c = a * b;
return c;
""",
    },
    {
        "id": "wrap",
        "name": "Wrap around",
        "description": "7 is the largest 4-bit value, so adding 1 wraps to -8.",
        "source": """a = 7;
b = a + 1;
return b;
""",
    },
    {
        "id": "compare",
        "name": "Signed comparison",
        "description": "-2 < 1 is true, though as unsigned bits 14 is larger than 1.",
        "source": """a = -2;
b = 1;
c = a < b;
return c;
""",
    },
    {
        "id": "equality",
        "name": "Equality",
        "description": "Equality needs no sign: it compares every bit.",
        "source": """a = -4;
b = 2;
c = b - 6;
d = a == c;
return d;
""",
    },
    {
        "id": "booleans",
        "name": "Boolean operators",
        "description": "Comparisons give booleans (bit 0 of a register), combined with &&, || and !.",
        "source": """a = -1;
b = 3;
c = (a < 0) && (b > 0);
d = (a > 0) || !(b > 0);
e = c && !d;
return e;
""",
    },
    {
        "id": "in-place",
        "name": "In-place updates",
        "description": "x += y and x -= 2 change the register where it sits: no copy and no new register.",
        "source": """x = 5;
y = -3;
x += y;
x -= 2;
return x;
""",
    },
    {
        "id": "in-place-branch",
        "name": "In-place update in a branch",
        "description": "An update under an if is a controlled add; the other branch leaves the register alone.",
        "source": """x = 2;
c = x < 5;
if c {
    x += 4;
}
x -= 1;
return x;
""",
    },
    {
        "id": "if-else",
        "name": "If and else",
        "description": "Both branches are always in the circuit, controlled on the condition and its negation.",
        "source": """a = -3;
b = 2;
c = a < b;
r = 0;
if c {
    r = b - a;
} else {
    r = a - b;
}
return r;
""",
    },
    {
        "id": "nested",
        "name": "Nested conditionals",
        "description": "Controls accumulate: an inner if is controlled on both conditions.",
        "source": """a = -2;
b = 3;
r = 0;
c = a < 0;
if c {
    d = b > 2;
    if d {
        r = a + b;
    } else {
        r = a - b;
    }
} else {
    r = a * b;
}
return r;
""",
    },
    {
        "id": "loop",
        "name": "A loop of updates",
        "description": "The loop is unrolled, so each iteration is one more in-place update.",
        "source": """s = 0;
for i in 0 .. 5 {
    s += i - 2;
}
return s;
""",
    },
    {
        "id": "calls",
        "name": "Calls",
        "description": "Calls are inlined; a parameter the callee updates is copied, so the caller's variable is untouched.",
        "source": """function bump(p: int, q: int): int {
    p += q;
    return p;
}

x = -2;
y = bump(x, 3);
z = bump(y, x);
return z;
""",
    },
    {
        "id": "mixed",
        "name": "Everything together",
        "description": "Negatives, a product, a comparison, an in-place update and a branch in one program.",
        "source": """a = -3;
b = 2;
p = a * b;
c = p < a;
acc = 0;
if c {
    acc += p;
} else {
    acc -= p;
}
acc += b;
return acc;
""",
    },
]
