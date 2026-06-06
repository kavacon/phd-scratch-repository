theory Chapter2
  imports Main
begin

value "1 + (2::nat)"
value "1 + (2::int)"
value "1 - (2::nat)" (* produces 0 since it is constrained by N *)
value "1 - (2::int)"

fun add:: "nat \<Rightarrow> nat \<Rightarrow> nat" where
  "add 0 n = n" |
  "add (Suc m) n = Suc(add m n)"

lemma add_zero[simp]: "add a  0 = a"
  apply(induction a)
  apply(auto)
  done

lemma add_one[simp]: "add m 1 = Suc m"
  apply(induction m)
  apply(auto)
  done

lemma add_succ[simp]: "add a (Suc c) = Suc (add a c)"
  apply(induction a)
  apply(auto)
  done

theorem add_com[simp]: "add a c  = add c a"
   apply(induction a)
   apply(auto)
   done

theorem add_ass_nat[simp]: "add a (add b c)  = add (add a b) c"
  apply(induction a)
  apply(auto)
  done

fun count:: "'a \<Rightarrow> 'a list \<Rightarrow> nat" where
  "count x [] = 0" |
  "count x (y # xs) = count x xs + (if x = y then 1 else 0)"


theorem count_lte_len[simp]: "count x xs \<le> length xs"
  apply(induction xs)
  apply(auto)
  done

fun snoc:: "'a list \<Rightarrow> 'a \<Rightarrow> 'a list" where
  "snoc [] x = [x]" |
  "snoc (a # as) x = a # (snoc as x)"

fun reverse:: "'a list \<Rightarrow> 'a list" where
  "reverse [] = []" |
  "reverse (x # xs) = snoc (reverse xs) x"


lemma snoc_holds[simp]: "snoc xs a = xs@[a]"
  apply(induction xs)
  apply(auto)
  done

lemma snoc_rev[simp]: "snoc (reverse xs) x = (reverse xs) @ [x]"
  apply(induction xs)
  apply(auto)
  done

lemma rev[simp]: "reverse (xs @ [x]) = x # reverse xs"
  apply(induction xs)
   apply(auto)
  done
  

theorem rev_rev[simp]: "reverse (reverse xs) = xs"
  apply(induction xs)
   apply(auto)
  done


fun sum_upto:: "nat \<Rightarrow> nat" where 
  "sum_upto 0 = 0" |
  "sum_upto n = n + sum_upto (n - 1)"


theorem sum_upto_n: "sum_upto n = (n * (n + 1)) div 2"
  apply(induction n)
  apply(auto)
  done

datatype 'a tree = Tip | Node "'a tree" 'a "'a tree"

fun contents:: "'a tree \<Rightarrow> 'a list" where
  "contents Tip = []" |
  "contents (Node l c r) = c # ((contents l) @ (contents r))"

fun sum_tree:: "nat tree \<Rightarrow> nat" where
  "sum_tree Tip = 0" |
  "sum_tree (Node l c r) = c + (sum_tree l) + (sum_tree r)"

theorem sum_tree_sum_list: "sum_tree t = sum_list (contents t)"
  apply(induction t)
  apply(auto)
  done

fun pre_order:: "'a tree \<Rightarrow> 'a list" where
  "pre_order Tip = []" |
  "pre_order (Node l c r) = c # ((pre_order l) @ (pre_order r))"

fun post_order:: "'a tree \<Rightarrow> 'a list" where
  "post_order Tip = []" |
  "post_order (Node l c r) = (post_order l) @ (post_order r) @ [c]"

fun mirror :: "'a tree \<Rightarrow>'a tree" where
  "mirror Tip= Tip" |
  "mirror (Node l a r) = Node (mirror r) a (mirror l)"

theorem pre_post_rev: "pre_order (mirror t) = rev (post_order t)"
  apply(induction t)
   apply(auto)
  done

fun intersperse:: "'a \<Rightarrow> 'a list \<Rightarrow> 'a list" where
  "intersperse x [] = []" |
  "intersperse a (x#xs) = [a, x] @ (intersperse a xs)"

theorem map_intersperse: "map f (intersperse a xs) = intersperse (f a) (map f xs)"
  apply(induction xs)
   apply(auto)
  done

fun itadd:: "nat \<Rightarrow> nat \<Rightarrow> nat" where
  "itadd 0 n = n" |
  "itadd (Suc m) n = itadd m (Suc n)"


theorem itadd_add: "itadd m n = add m n"
  apply(induction m arbitrary: n)
   apply(auto)
  done

datatype tree0 = Leaf | Node tree0 tree0

fun nodes:: "tree0 \<Rightarrow> nat" where
  "nodes Leaf = 1" |
  "nodes (Node l r) = 1 + (nodes l) + (nodes r)"

fun explode :: "nat \<Rightarrow> tree0 \<Rightarrow> tree0" where
  "explode 0 t = t" |
  "explode (Suc n) t = explode n (Node t t)"


theorem len_explode: "nodes (explode n t) = 2^n*(nodes t + 1) - 1"
  apply(induction n arbitrary: t)
  apply(auto simp add: algebra_simps)
  done

datatype exp = Var | Const int | Add exp exp | Mult exp exp

fun eval:: "exp \<Rightarrow> int \<Rightarrow> int" where
  "eval Var x = x" |
  "eval (Const y) x = y" |
  "eval (Add e1 e2) x = (eval e1 x) + (eval e2 x)" |
  "eval (Mult e1 e2) x = (eval e1 x) * (eval e2 x)"

fun evalp:: "int list \<Rightarrow> int \<Rightarrow> int" where
  "evalp [] n = 0" |
  "evalp (c #cs) x = c + x*(evalp cs x)"

fun add_coeffs:: "int list \<Rightarrow> int list \<Rightarrow> int list" where
  "add_coeffs [] xs = xs" |
  "add_coeffs xs [] = xs" |
  "add_coeffs (x#xs) (y#ys) = (x + y) # (add_coeffs xs ys)"

fun mult_coeffs:: "int list \<Rightarrow> int list \<Rightarrow> int list" where
  "mult_coeffs [] xs = []" |
  "mult_coeffs xs [] = []" |
  "mult_coeffs [x] (y#ys) = (x*y) # (mult_coeffs [x] ys)" |
  "mult_coeffs (y#ys) [x] = (x*y) # (mult_coeffs [x] ys)" |
  "mult_coeffs (x#xs) ys = add_coeffs (mult_coeffs [x] ys) (mult_coeffs xs (0# ys))"

fun coeffs:: "exp \<Rightarrow> int list" where
  "coeffs (Const y) = [y]" |
  "coeffs Var = [0,1]" |
  "coeffs (Add e1 e2) = add_coeffs(coeffs e1) (coeffs e2)" |
  "coeffs (Mult e1 e2) = mult_coeffs (coeffs e1) (coeffs e2)"

lemma evalp_add_coeffs[simp]: "evalp (add_coeffs xs ys) x = evalp xs x + evalp ys x"
  apply(induction xs ys  arbitrary: x rule: add_coeffs.induct)
  apply(auto simp add: algebra_simps)
  done

lemma evalp_mult_coeffs[simp]: "evalp (mult_coeffs xs ys) x = evalp xs x * evalp ys x"
  apply(induction xs ys  arbitrary: x rule: mult_coeffs.induct)
  apply(auto simp add: algebra_simps)
  done

theorem evalp_coeffs: "evalp (coeffs e) x = eval e x" 
  apply(induction e arbitrary: x)
     apply(auto simp add: algebra_simps)
  done
  
