theory Chapter4
  imports Main IMP
begin

datatype 'a tree = Tip | Node "'a tree" 'a "'a tree"

fun set:: "'a tree \<Rightarrow> 'a set" where
  "set Tip = {}" |
  "set (Node l c r) = set(l) Un {c} Un set(r)"

fun ord:: "int tree \<Rightarrow> bool" where
  "ord Tip = True" |
  "ord (Node l c r) = ((\<forall>y \<in> set l. y \<le> c) \<and> (\<forall>z \<in> set r. c \<le> z) \<and> ord l \<and> ord r)"

(* you are writing proofs not code *)
fun ins:: "int tree \<Rightarrow> int \<Rightarrow> int tree" where
  "ins Tip x = Node Tip x Tip" |
  "ins (Node l c r) x = (if x = c then (Node l c r) else if x < c then Node (ins l x) c r else Node l c (ins r x))"

lemma [simp]: "set (ins t x) = {x} Un set(t)"
  apply(induction t arbitrary: x)
   apply(auto)
  done

lemma "ord t \<Longrightarrow> ord (ins t i)"
  apply(induction t arbitrary: i)
   apply(auto)
  done

inductive palindrome:: "'a list \<Rightarrow> bool" where
  pal_empty: "palindrome []" |
  pal_element: "palindrome xs \<Longrightarrow> palindrome (a # xs @ [a])"

lemma palindrome_rev: "palindrome xs \<Longrightarrow> rev xs = xs"
  apply(induction rule: palindrome.induct)
   apply(auto)
  done

inductive star' :: "('a \<Rightarrow>'a \<Rightarrow> bool) \<Rightarrow>'a \<Rightarrow>'a \<Rightarrow>bool" for r where
  refl': "star' r x x" |
  step': "star' r x y \<Longrightarrow> r y z \<Longrightarrow>  star' r x z"

inductive star :: "('a \<Rightarrow>'a \<Rightarrow> bool) \<Rightarrow>'a \<Rightarrow>'a \<Rightarrow>bool" for r where
  refl: "star r x x" |
  step: "star r x y \<Longrightarrow> star r y z \<Longrightarrow>  star r x z"

lemma star_trans[simp]: "star r x y \<Longrightarrow> star r y z \<Longrightarrow> star r x z"
  apply(induction rule: star.induct)
   apply(assumption)
  apply(metis)
  done



lemma star'_trans[simp]: "star r x y \<Longrightarrow> star' r y z \<Longrightarrow> star' r x z"
  apply(induction rule: star.induct)
   apply(auto)
  done
  
  
lemma star_star': "star r x y \<Longrightarrow> star' r x y"
  apply(induction rule: star.induct)
   apply(rule refl')
  apply(auto)
  done

datatype alpha = a | b

inductive S :: "alpha list \<Rightarrow> bool" where
 S_empty:  "S []" |
 S_bound: "S w \<Longrightarrow> S(a # w @ [b]) " |
 S_dupe: "S w1 \<Longrightarrow> S w2 \<Longrightarrow> S (w1 @ w2)"

inductive T :: "alpha list \<Rightarrow> bool" where
 T_empty:  "T []" |
 T_bound: "T w1 \<Longrightarrow> T w2 \<Longrightarrow> T(w1 @ a #  w2 @ [b])"

lemma T_S[simp]: "T w \<Longrightarrow> S w"
  apply(induction rule: T.induct)
   apply(rule S_empty)
  apply(simp add: S_bound S_dupe)
  done

lemma T_concat[simp]: "T w2 \<Longrightarrow> T w1 \<Longrightarrow> T (w1 @ w2)"
  apply(induction rule: T.induct)
   apply(simp)
  apply(metis T_bound append_assoc)
  done

lemma S_T[simp]: "S w \<Longrightarrow> T w"
  apply(induction rule: S.induct)
    apply(rule T_empty)
  apply(subst append_Nil[symmetric])
   apply(metis T_bound T_empty)
  apply(auto)
  done

lemma S_equal_T: "S w = T w"
  by(metis S_T T_S)

(*
fun aval :: "aexp \<Rightarrow>state \<Rightarrow> val" where
  "aval (N n) s = n" |
  "aval (V x) s = s x" |
  "aval (Plus a1 a2) s = aval a1 s + aval a2 s"
*)

inductive aval_rel:: "aexp \<Rightarrow> state \<Rightarrow> val \<Rightarrow> bool" where
  aval_const: "aval_rel (N n) s n" |
  aval_var: "aval_rel (V v) s (s v)" |
  aval_plus: "aval_rel a1 s x1 \<Longrightarrow> aval_rel a2 s x2 \<Longrightarrow> aval_rel (Plus a1 a2) s (x1 + x2)"

lemma aval_rel_ind[simp]: "aval_rel e s v \<Longrightarrow> aval e s = v"
  apply(induction rule: aval_rel.induct)
    apply(auto)
  done

lemma aval_ind_rel[simp]: "aval e s = v \<Longrightarrow> aval_rel e s v"
  apply(induction e arbitrary: s v)
  apply(auto intro: aval_const aval_var aval_plus)
  done

lemma "aval_rel e s v = ( aval e s = v)"
  by (metis aval_rel_ind aval_ind_rel)

inductive ok :: "nat \<Rightarrow> instr list \<Rightarrow> nat \<Rightarrow> bool" where
  ok_empty: "ok n [] n" |
  ok_loadi: "ok (n+1) is m \<Longrightarrow> ok n (LOADI x # is) m" |
  ok_load: "ok (n+1) is m \<Longrightarrow> ok n (LOAD x # is) m" |
  ok_add: "n \<ge> 2 \<Longrightarrow> ok (n-1) is m \<Longrightarrow> ok n (ADD # is) m"

fun exec14 :: "instr \<Rightarrow> state \<Rightarrow> stack \<Rightarrow> stack" where
  "exec14 (LOADI n) _ stk = (n # stk)" |
  "exec14 (LOAD x) s stk = (s(x) # stk)" |
  "exec14 ADD _ (j # i # stk) = ( (i + j) # stk)"

fun exec4 :: "instr list \<Rightarrow> state \<Rightarrow> stack \<Rightarrow> stack" where
  "exec4 [] _ stk = stk" |
  "exec4 (i#is) s (stk) = exec4 is s (exec14 i s stk)"

lemma ok_stack[simp]: "\<lbrakk>ok n is n';  length stk = n \<rbrakk> \<Longrightarrow> length (exec4 is s stk) = n'"
  apply(induction arbitrary: stk s rule: ok.induct)
  apply(cases stk)
      apply(auto)


  