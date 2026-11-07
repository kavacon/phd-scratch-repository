"""
Programs for measuring the source-level rewrites: patterns the literature reports as costly or lossy, written in the
language. Each is measured as it stands (`WORKLOADS`), and where a rewrite by hand is known to keep the outcomes the same, with
the rewritten source beside it (`REWRITES`), so the cost of a rewrite can be read off before it is built. These are for
measurement; the intended outcomes of programs are in `tests/golden.py`.

Each workload is (family, name, source, literature pattern). Each rewrite is (rewrite, workload name, rewritten source).
"""

WORKLOADS = [
 # nesting of conditions on a distribution (Spire P1, SPARE conditionally clean ancillae)
 ("nest","depth 2","a=0..4;b=0..4;x=0; if a<2 { if b<2 { x+=1; } } return x;","Spire nested conditionals"),
 ("nest","depth 3","a=0..4;b=0..4;c=0..4;x=0; if a<2 { if b<2 { if c<2 { x+=1; } } } return x;","Spire"),
 ("nest","depth 4","a=0..4;b=0..4;c=0..4;d=0..4;x=0; if a<2 { if b<2 { if c<2 { if d<2 { x+=1; } } } } return x;","Spire"),
 ("nest","depth 5","a=0..4;b=0..4;c=0..4;d=0..4;e=0..4;x=0; if a<2 { if b<2 { if c<2 { if d<2 { if e<2 { x+=1; } } } } } return x;","Spire"),
 ("nest","depth 4 with else","a=0..4;b=0..4;c=0..4;d=0..4;x=0; if a<2 { if b<2 { if c<2 { if d<2 { x+=1; } else { x+=2; } } } } return x;","Spire"),
 ("nest","loop of guarded updates","a=0..4;x=0; for i in 0..4 { if a>i { x+=1; } } return x;","threshold counter, SPARE conditional MCS"),
 ("nest","loop of nested guards","a=0..4;b=0..4;x=0; for i in 0..3 { if a>i { if b>i { x+=1; } } } return x;","control per layer"),
 ("nest","same guard twice (fusable)","a=0..4;b=0..4;x=0;y=0; if a<b { x+=1; } if a<b { y+=2; } return x+y;","SPARE: consecutive CTU blocks"),
 ("nest","guard of a guard (nested, same inner)","a=0..4;b=0..4;x=0; if a<b { if a<b { x+=1; } } return x;","merge conditionals"),
 # late distributions / work between creation and use (P15)
 ("late","distribution made early, used last","a=0..8;b=0..4;y=0; for i in 0..4 { y+=b; } return y+a;","P15"),
 ("late","two made early, used last","a=0..8;c=0..8;b=0..4;y=0; for i in 0..4 { y+=b; } return y+a+c;","P15"),
 ("late","made early, dead branch between","a=0..8;b=0..4;y=0; for i in 0..3 { y+=b; y+=1; } return a;","P15"),
 ("late","range that is always true","a=0..4;x=0; if a<10 { x=1; } return x;","values that look superposed"),
 ("late","product with zero","a=0..4;b=a*0+3; return b;","values that look superposed"),
 ("late","difference of itself added","a=0..4;b=(a-a)+3; return b+(a-a);","values that look superposed"),
 # arm work (P14)
 ("arms","common product in both arms","a=0..4;b=0..4;x=0; if a<2 { x=b*b+1; } else { x=b*b+2; } return x;","P14 common work"),
 ("arms","arms differ by a constant","a=0..4;x=0; if a<2 { x=5; } else { x=7; } return x;","P14 arms to arithmetic"),
 ("arms","arms differ by a constant, distribution operand","a=0..4;b=0..4;x=0; if a<2 { x=b+5; } else { x=b+7; } return x;","P14"),
 ("arms","shared work across loop","b=0..4;x=0; for i in 0..4 { x+=b*b; } return x;","P14 repeated computation"),
 ("arms","same call twice","function f(p) { return p*p+p; }\nb=0..4; return f(b)+f(b);","P14 shared across calls"),
 ("arms","call in both arms","function f(p) { return p*p+p; }\na=0..4;b=0..4;x=0; if a<2 { x=f(b); } else { x=f(b)+1; } return x;","P14 / Unqomp modularity"),
 # collapsing and moves (P11, P13)
 ("chain","chain of temporaries","a=0..4;y=a+1;z=y+1;w=z+1; return w;","variable collapsing"),
 ("chain","chain of temporaries, source dead","a=0..4;y=a+1;z=y+1;w=z+1; return w;","copy to move (a dead after first use)"),
 ("chain","reassignment chain","a=0..4;x=a+1; x+=2; x+=3; x+=4; return x;","variable collapsing"),
 ("chain","temporaries of distinct operands","a=0..4;b=0..4;t=a+b;u=t+a;v=u+b; return v;","Reqomp linear dependency"),
 ("chain","long linear chain","a=0..2;b=0..2;c=0..2;d=0..2;e=0..2;t1=a+b;t2=t1+c;t3=t2+d;t4=t3+e; return t4;","Reqomp pebbling chain"),
 ("chain","long linear chain, each value used twice","a=0..2;b=0..2;t1=a+b;t2=t1+a;t3=t2+t1;t4=t3+t2; return t4;","Reqomp / Bennett"),
 # information loss and uncomputation limits (P3, P4, P12)
 ("loss","condition on the variable it updates","x=0..4; if x<2 { x+=1; } return x;","Qrisp in-place under condition computed from it"),
 ("loss","xor feedback","a=0..4;f=a<2; a+=1; g=a<2; return f&&g;","Reqomp: value used after its source changed"),
 ("loss","even multiple","a=0..4; c=a*2; return c;","P12 information loss"),
 ("loss","square of a sum","a=0..4;b=0..4; return (a+b)*(a+b);","shared operand register"),
 ("loss","comparison chain","a=0..4;b=0..4;c=0..4;d=0..4; return (a<b)&&(b<c)&&(c<d);","comparator garbage"),
 ("loss","oracle: sum of squares equals constant","a=0..4;b=0..4; return (a*a+b*b)==5;","Grover predicate, compute-transform-uncompute"),
 ("loss","oracle inside a guard","a=0..4;b=0..4;x=0; if (a*a+b*b)==5 { x+=1; } return x;","guard from a predicate"),
 ("loss","overwrite a dist with a function of itself","a=0..4; a+=a; a+=a; return a;","self update"),
 # nested calls (SQUARE recomputation, Unqomp modularity)
 ("calls","three levels of calls","function h(p) { return p+1; }\nfunction g(p) { return h(p)*2; }\nfunction f(p) { return g(p)+h(p); }\na=0..4; return f(a);","SQUARE nested calls"),
 ("calls","three levels of calls under a guard","function h(p) { return p+1; }\nfunction g(p) { return h(p)*2; }\nfunction f(p) { return g(p)+h(p); }\na=0..4;b=0..4;x=0; if b<2 { x=f(a); } return x;","SQUARE reclaim under condition"),
 ("calls","call in a loop under a guard","function sq(p) { return p*p; }\na=0..4;b=0..3;x=0; for i in 0..3 { if b>i { x+=sq(a); } } return x;","recursion-like layers"),
]

REWRITES = [
 ("flatten","depth 3","a=0..4;b=0..4;c=0..4;x=0; if (a<2)&&(b<2)&&(c<2) { x+=1; } return x;"),
 ("flatten","depth 5","a=0..4;b=0..4;c=0..4;d=0..4;e=0..4;x=0; if (a<2)&&(b<2)&&(c<2)&&(d<2)&&(e<2) { x+=1; } return x;"),
 ("flatten","depth 4 with else","a=0..4;b=0..4;c=0..4;d=0..4;x=0; if (a<2)&&(b<2)&&(c<2) { if d<2 { x+=1; } else { x+=2; } } return x;"),
 ("merge","same guard twice (fusable)","a=0..4;b=0..4;x=0;y=0; if a<b { x+=1; y+=2; } return x+y;"),
 ("merge","guard of a guard (nested, same inner)","a=0..4;b=0..4;x=0; if a<b { x+=1; } return x;"),
 ("late","distribution made early, used last","b=0..4;y=0; for i in 0..4 { y+=b; } a=0..8; return y+a;"),
 ("late","range that is always true","x=1; return x;"),
 ("late","two made early, used last","b=0..4;y=0; for i in 0..4 { y+=b; } a=0..8;c=0..8; return y+a+c;"),
 ("arms","common product in both arms","a=0..4;b=0..4;x=b*b+1; if a>1 { x+=1; } return x;"),
 ("arms","arms differ by a constant","a=0..4;x=5; if a>1 { x+=2; } return x;"),
 ("arms","arms differ by a constant, distribution operand","a=0..4;b=0..4;x=b+5; if a>1 { x+=2; } return x;"),
 ("arms","shared work across loop","b=0..4;t=b*b;x=0; for i in 0..4 { x+=t; } return x;"),
 ("arms","call in both arms","function f(p) { return p*p+p; }\na=0..4;b=0..4;x=f(b); if a>1 { x+=1; } return x;"),
 ("chain","chain of temporaries","a=0..4;w=a+3; return w;"),
 ("chain","reassignment chain","a=0..4;x=a+10; return x;"),
 ("chain","long linear chain","a=0..2;b=0..2;c=0..2;d=0..2;e=0..2; return a+b+c+d+e;"),
]
