"""Programs preloaded in the demo. Each one runs, so the demo can check its result and its registers."""

PROGRAMS = [
    {
        "id": "area",
        "name": "A temporary",
        "description": "p only exists to help compute q, so area releases it. w and h belong to the caller, so area leaves them alone and the caller releases x and y once the call is done.",
        "source": """function area(w: int, h: int): int {
    p = w * h;
    q = p + 1;
    return q;
}

x = 3;
y = 4;
return area(x, y);
""",
    },
    {
        "id": "chain",
        "name": "A chain of temporaries",
        "description": "b is computed from a, so b has to be released before a. Releases run in reverse order of creation.",
        "source": """function f(x: int): int {
    a = x + 1;
    b = a * 2;
    c = b + a;
    return c;
}

return f(3);
""",
    },
    {
        "id": "expression",
        "name": "Returning an expression",
        "description": "The returned expression is named first, so the values it reads become temporaries that can be released.",
        "source": """function f(a: int): int {
    t = a + 1;
    u = t * 3;
    return t + u;
}

return f(2);
""",
    },
    {
        "id": "if-else",
        "name": "If / else",
        "description": "Both branches define y#2, so only one ever runs. Its release is an if on the same condition.",
        "source": """function f(x: int): int {
    a = x + 1;
    y = 0;
    if a < 5 {
        t = a * 2;
        y = t + 1;
    } else {
        y = a - 1;
    }
    r = y + 1;
    return r;
}

return f(2);
""",
    },
    {
        "id": "one-armed",
        "name": "An if without an else",
        "description": "Balance adds the missing else (x = x;) so x has a new value on every path.",
        "source": """function f(a: int): int {
    x = a + 1;
    if x < 4 {
        x = x + 5;
    }
    return x;
}

return f(1);
""",
    },
    {
        "id": "nested",
        "name": "Nested ifs",
        "description": "The release of x#4 nests the same way the ifs do, with each branch's definitions substituted in.",
        "source": """function f(a: int): int {
    x = a;
    if a < 5 {
        if a < 3 {
            x = a + 1;
        } else {
            x = a + 2;
        }
        x = x * 2;
    } else {
        x = a - 1;
    }
    r = x + a;
    return r;
}

return f(1);
""",
    },
    {
        "id": "loop",
        "name": "A loop",
        "description": "Temporaries inside the body are released every iteration. The value carried round the loop is not released.",
        "source": """function g(n: int): int {
    s = 0;
    for i in 0 .. n {
        t = s + i;
        s = t * 2 + 1;
    }
    return s;
}

return g(3);
""",
    },
    {
        "id": "width",
        "name": "Early release lowers peak width",
        "description": "a and b only feed y, so they are released right after y is built, before the later scratch work needs registers.",
        "source": """function f(x: int): int {
    a = x + 1;
    b = a * 2;
    y = b + 3;
    // scratch work that does not touch a, b or y
    c = x + 5;
    d = c * 3;
    e = d + 7;
    return y;
}

return f(2);
""",
    },
    {
        "id": "explicit",
        "name": "A hand-written release",
        "description": "t is released by hand with ~=. The pass leaves it alone and only adds the releases that are missing.",
        "source": """function f(a: int): int {
    t = a + 1;
    u = t * 2;
    v = u + t;
    t ~= a + 1;
    return v;
}

return f(2);
""",
    },
    {
        "id": "calls",
        "name": "Functions calling functions",
        "description": "Each function is analysed on its own.",
        "source": """function double(x: int): int {
    y = x * 2;
    return y + 0;
}

function quad(x: int): int {
    d = double(x);
    return double(d);
}

return quad(3);
""",
    },
]
