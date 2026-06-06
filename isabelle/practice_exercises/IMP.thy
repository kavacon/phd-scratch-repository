theory IMP
  imports Main
begin

type_synonym vname = string
datatype aexp= N int | V vname | Plus aexp aexp
type_synonym val= int
type_synonym state = "vname \<Rightarrow> val"
datatype instr = LOADI val | LOAD vname | ADD
type_synonym stack= "val list"

fun plus :: "aexp \<Rightarrow>aexp \<Rightarrow>aexp" where
  "plus (N i1) (N i2) = N(i1+i2)" |
  "plus (N i) a = (if i=0 then a else Plus (N i) a)" |
  "plus a (N i) = (if i=0 then a else Plus a (N i))" |
  "plus a1 a2 = Plus a1 a2"

fun asimp :: "aexp \<Rightarrow>aexp" where
  "asimp (N n) = N n" |
  "asimp (V x) = V x" |
  "asimp (Plus a1 a2) = plus (asimp a1) (asimp a2)"

fun asimp_const :: "aexp \<Rightarrow>aexp" where
  "asimp_const (N n) = N n" |
  "asimp_const (V x) = V x" |
  "asimp_const (Plus a1 a2) =
    (case (asimp_const a1, asimp_const a2) of
      (N n1, N n2) \<Rightarrow> N(n1+n2) |
      (b1,b2) \<Rightarrow> Plus b1 b2)"

fun optimal:: "aexp \<Rightarrow> bool" where
  "optimal (N n) = True" |
  "optimal (V x) = True" |
  "optimal (Plus (N n) (N m)) = False" |
  "optimal (Plus e1 e2) = conj (optimal e1) (optimal e2)"

theorem asimp_optimal: "optimal(asimp_const a)"
  apply(induction a rule: asimp_const.induct)
    apply(auto split: aexp.split)
  done

fun full_plus:: "aexp \<Rightarrow> aexp \<Rightarrow> aexp" where
  "full_plus (N n1) (Plus a (N n2)) = Plus a (N (n1 + n2))" |
  "full_plus (N n1) (Plus (N n2) a) = Plus a (N (n1 + n2))" |
  "full_plus (Plus a (N n2)) (N n1) = Plus a (N (n1 + n2))" |
  "full_plus (Plus (N n2) a) (N n1) = Plus a (N (n1 + n2))" |
  "full_plus (Plus a1 (N n1)) (Plus a2 (N n2)) = Plus (full_plus a1 a2) (N (n1 + n2))" |
  "full_plus a1 a2 = plus a1 a2"

fun full_asimp:: "aexp \<Rightarrow> aexp" where 
  "full_asimp (N n) = N n" |
  "full_asimp (V v) = V v" |
  "full_asimp (Plus a1 a2) = full_plus (full_asimp a1) (full_asimp a2)"

fun aval :: "aexp \<Rightarrow>state \<Rightarrow> val" where
  "aval (N n) s = n" |
  "aval (V x) s = s x" |
  "aval (Plus a1 a2) s = aval a1 s + aval a2 s"

lemma aval_plus [simp]: "aval (plus a1 a2) s = aval a1 s + aval a2 s"
  apply(induction a1 a2 rule: plus.induct)
              apply(auto)
  done

lemma aval_full_plus[simp]: "aval (full_plus a1 a2) s = aval a1 s + aval a2 s"
  apply(induction a1 a2 arbitrary: s rule: full_plus.induct)
                      apply(auto simp: algebra_simps split: aexp.split)
  done
  
theorem "aval (full_asimp a) s = aval a s"
  apply(induction a arbitrary: s rule: full_asimp.induct)
    apply(auto simp: algebra_simps split: aexp.split)
  done

fun subst:: "vname \<Rightarrow> aexp \<Rightarrow> aexp \<Rightarrow> aexp" where
  "subst v e (N n) = N n" |
  "subst v e (V w) = (if w = v then e else V w)" |
  "subst v e (Plus a1 a2) = Plus (subst v e a1) (subst v e a2)"

lemma subst_lemma[simp]: "aval (subst x a e) s = aval e (s(x := aval a s))"
  apply(induction e arbitrary: x a s)
    apply(auto simp: algebra_simps split: aexp.split)
  done

theorem equal_exp: "aval a1 s = aval a2 s \<Longrightarrow> aval (subst x a1 e) s = aval (subst x a2 e) s"
  apply(induction e arbitrary: s x a1 a2)
    apply(auto)
  done

fun exec1 :: "instr \<Rightarrow> state \<Rightarrow> stack \<Rightarrow> stack option" where
  "exec1 (LOADI n) _ stk = Some (n # stk)" |
  "exec1 (LOAD x) s stk = Some (s(x) # stk)" |
  "exec1 ADD _ (j # i # stk) = Some( (i + j) # stk)" |
  "exec1 ADD _ [] = None" |
  "exec1 ADD _ [j] = None"

fun exec :: "instr list \<Rightarrow> state \<Rightarrow> stack option \<Rightarrow> stack option" where
  "exec [] _ stk = stk" |
  "exec (i#is) s (Some stk) = exec is s (exec1 i s stk)" |
  "exec (i#is) s None = None"

fun comp :: "aexp \<Rightarrow> instr list" where
  "comp (N n) = [LOADI n]" |
  "comp (V x) = [LOAD x]" |
  "comp (Plus e1 e2) = comp e1 @ comp e2 @ [ADD]"

lemma [simp]:  "exec (is1 @ is2) s stk = exec is2 s (exec is1 s stk)"
  apply(induction is1 arbitrary: s stk)
   apply(auto split: option.splits)
  apply(case_tac stk)
  apply(induction is2)
    apply(auto)
  done

lemma [simp]: "exec (comp a) s (Some stk) = Some ( (aval a s) # stk)"
  apply(induction a arbitrary: s stk)
    apply(auto)
  done

type_synonym reg = "nat"
datatype instr1 = LDI int reg | LD vname reg | ADD1 reg reg

fun execr1 :: "instr1 \<Rightarrow> state \<Rightarrow> (reg \<Rightarrow> int) \<Rightarrow> (reg \<Rightarrow> int)" where
  "execr1 (LDI i r) s m = m(r:= i)" |
  "execr1 (LD v r) s m = m(r:= s(v))" |
  "execr1 (ADD1 r1 r2) s m = m(r1:= m(r1) + m(r2))"

fun compr :: "aexp \<Rightarrow> reg \<Rightarrow> instr1 list" where
  "compr (N n) r = [LDI n r]" |
  "compr (V x) r = [LD x r]" |
  "compr (Plus e1 e2) r = compr e1 r @ compr e2 (r+1) @ [ADD r (r+1)]"

fun execr :: "instr1 list \<Rightarrow> state \<Rightarrow> (reg \<Rightarrow> int) \<Rightarrow> (reg \<Rightarrow> int)" where
    "execr [] _ m = m" |
    "execr (i#is) s m = execr is s (execr1 i s m)"

lemma [simp]:  "execr (is1 @ is2) s rs r = execr is2 s (execr is1 s rs) r"
  apply(induction is1 arbitrary: s rs r)
   apply(auto)
  done


lemma compr_reg_preserved[simp]: "r < r' \<Longrightarrow> execr (compr a r') s rs r = rs r"
  apply(induction a arbitrary: r' s rs)
    apply(auto)
  done

lemma "execr (compr a r) s rs r = aval a s"
  apply(induction a arbitrary: rs s r)
    apply(auto simp: algebra_simps)
  done

datatype instr0 = LDI0 val | LD0 vname | MV0 reg | ADD0 reg

fun exec01 :: "instr0 \<Rightarrow> state \<Rightarrow> (reg \<Rightarrow> int) \<Rightarrow> (reg \<Rightarrow> int)" where
  "exec01 (LDI0 i) s m = m(0:= i)" |
  "exec01 (LD0 v) s m = m(0:= s(v))" |
  "exec01 (ADD0 r) s m = m(0:= m(0) + m(r))" |
  "exec01 (MV0 r) s m = m(r:=  m(0))"

fun comp0 :: "aexp \<Rightarrow> reg \<Rightarrow>  instr0 list" where
  "comp0 (N n) _  = [LDI0 n]" |
  "comp0 (V x) _ = [LD0 x]" |
  "comp0 (Plus e1 e2) r = comp0 e1 (r+1) @ [MV0 r] @ comp0 e2 (r+1)  @ [ADD0 r]"

fun exec0 :: "instr0 list \<Rightarrow> state \<Rightarrow> (reg \<Rightarrow> int) \<Rightarrow> (reg \<Rightarrow> int)" where
    "exec0 [] _ m = m" |
    "exec0 (i#is) s m = exec0 is s (exec01 i s m)"

lemma [simp]: "exec0 (is1 @ is2) s rs = exec0 is2 s (exec0 is1 s rs)"
  apply(induction is1 arbitrary: s rs)
  apply(auto)
  done

lemma [simp]: "0 < r \<Longrightarrow> r < r' \<Longrightarrow> exec0 (comp0 a r') s rs r  = rs r"
  apply(induction a arbitrary: r' s rs)
    apply(auto)
  done

(*
lemma [simp]: "exec0 (comp0 a r) s rs 0 = aval a s"
  apply(induction a arbitrary: r s rs)
    apply(auto simp: algebra_simps)
*)

end
