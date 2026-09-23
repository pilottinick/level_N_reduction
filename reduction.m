// This code implements the algorithm described in 
// "Reduction theory for binary quadratic forms with level"
// by Jennifer Johnson-Leung and Brooks Roberts, hereafter refered to as [JR].

// Let BQF(N) denote the set of all positive-definite integral binary quadratic forms a*x^2 + b*x*y + c*y^2 where 
// a, b, c are integers and such that a is divisible by N. We denote such a form by (a, b, c). For example, BQF(1) is 
// the set of all positive-definite integeral quadratic forms.

// There is a left action of the group \Gamma^0(N) on BQN(N). Let E denote an element of \Gamma^0(N)\BQN(N). An element
// (a_r, b_r, c_r) in E is said to be N-reduced (or simply reduced) if
// 1) a_r is minimal among all integers a such that there exists b and c with (a, b, c) in E and 
// 2) b_r is maximal among all integers b for which there exists an integer c with (a_r, b, c) in E and -a_r < b <= a_r.

intrinsic Eval(a::RngIntElt, b::RngIntElt, c::RngIntElt, x::RngIntElt, y::RngIntElt) -> RngIntElt
  {The value of the quadratic form (a, b, c) at (x, y).}

  return a*x^2 + b*x*y + c*y^2;
end intrinsic;

// An integeral quadratic form (a, b, c) is represented by a half-integral matrix M = [[a,b/2],[b/2,c]]. A unimodular 
// matrix u in GL_2(Z) acts on BQN(1) by its action on half-integral matrices [u]M = u * M * u^t. The image of (a, b, c)
// under u is denoted by u.(a,b,c).
intrinsic UnimodularAction(u::AlgMatElt, a::RngIntElt, b::RngIntElt, c::RngIntElt) -> RngIntElt, RngIntElt, RngIntElt
  {The image of the integral quadratic form (a, b, c) under the unimodular matrix u.}

  g1 := u[1][1]; g2 := u[1][2]; g3 := u[2][1]; g4 := u[2][2];
  return Eval(a, b, c, g1, g2), 2*a*g1*g3 + b*(g2*g3 + g1*g4) + 2*c*g2*g4, Eval(a, b, c, g3, g4);
end intrinsic;

// SReduce and TReduce are the basic operations used in the classical 1-reduction algorithm. These operations correspond
// to the generators S := [[0,-1],[1,0]] and T := [[1,1],[0,1]] of SL_2(Z).

// Define S to be the matrix [[0,-1],[1,0]]. We have S.(a, b, c) = (c, -b, a).
intrinsic SReduce(a, b, c) -> RngIntElt, RngIntElt, RngIntElt, AlgMatElt
  {Apply a power of S to (a, b, c) and output a 1-equivalent form (a_r, b_r, c_r) such that a_r <= c_r and b >= 0 if 
   a_r = c_r. Also output the power of S which is applied.}

  if a lt c or (a eq c and b ge 0)then
    return a, b, c, Matrix([[1,0],[0,1]]);
  else
    return c, -b, a, Matrix([[0,-1],[1,0]]);
  end if;
end intrinsic;

// We define T to be the matrix [[1,0],[1,1]]. We have T.(a, b, c) = (a, b + 2*a, c + b + a).
intrinsic TReduce(a, b, c) -> RngIntElt, RngIntElt, AlgMatElt
  {Apply a power of T to (a, b, c) and output a 1-equivalent form (a, b_r, c_r) such that -a < b_r <= a. Also output the
   power of T which is appied.}

  two_a := 2*a;
  m := (a - b) div two_a;
  c +:= m*b + m^2*a;
  b +:= m*two_a;
  return b, c, Matrix([[1,0],[m,1]]);
end intrinsic;

// The 1-reduction of an integral quadratic form (a,b,c) agrees with the classical notion of reduction due to Lagrange
// and Gauss. The following outputs the 1-reduction of an integral quadratic form (a,b,c).
intrinsic LagrangeReduce(a, b, c) -> RngIntElt, RngIntElt, RngIntElt, AlgMatElt
  {Compute the 1-reduction of a positive definite (a,b,c) and a matrix A in SL_2(Z) such that A.(a,b,c) is 1-reduced}

  a, b, c, A := SReduce(a, b, c);
  S := Matrix([[0,-1],[1,0]]);
  while S eq Matrix([[0,-1],[1,0]]) do
    b, c, T := TReduce(a, b, c);
    a, b, c, S := SReduce(a, b, c);
    A := S*T*A;
  end while;

  return a, b, c, A;
end intrinsic;

// Here, we overview of the series of reductions motivating the algorithm:
// 1) Finally, we wish to compute the set A(N, F) of forms (a_r, b_0, c_0) which are N-equivalent to F := (a,b,c) and 
//    such that a_r is minimal and -a_r < b_0 <= a_r. This is implemented below in NReduceA.
// 2) A(N, F) can be computed from the set R(N, F) of integers pairs (h_1, h_2) such that gcd(h_1, h_2) = 1, h_2 is 
//    divisible by N, and Eval(a, b, c, h_1, h_2) = a_r.
// 3) R(N, F) can be computed from a set K(N, g_1, g_3). We call this the set of "admissible vectors." To define
//    K(N, g_1, g_3), let F_1 := (a / N, b, c * N) denote the "twist" of (a, b, c), let F_0 denote the 1-reduction 
//    of F_1 and let the matrix g = [[g_1,g_2],[g_3,g_4]] be such that g.F_1 = F_0. Then K(N, g_1, g_3) is defined to be
//    the set of integers pairs (t_1, t_2) such that gcd(t_1, t_2) = 1 and gcd(t_1 * g_1 + t_2 * g_3, N) = 1. This is
//    implemented in AdmissibleVectors.
// 4) K(N, g_1, g_3) is computed by an application of the topograph. The topograph is implemented in the file 
//    topograph..m Strictly speaking, we compute the sets K(N, g_1, g_3, i) for i up to a sufficiently large finite bound.

// AdmissibleVectors computes the set K(N, g_1, g_3, i).
intrinsic AdmissibleVectors(N, g1, g3, i : T := Topograph()) -> SeqEnum
  {Output the list of admissible vectors for N, g1, and g_3.}
  K := [];
  for v in GetLaxVectors(T, i) do
    if Gcd(v[1]*g1 + v[2]*g3, N) eq 1 then
      Append(~K, v);
    end if;
  end for;

  return K;
end intrinsic;

// NReduceA computes the set A(n, F).
intrinsic NReduceA(a, b, c, N) -> SeqEnum
  {Compute the forms (a_r, b_0, c_0) which are N-equivalent to the level-N form (a,b,c) such that a_r is minimal and 
   -a_r < b_0 <= a_r. Output a list of such forms sorted in order of increasing values of b_0.}

  a0, b0, c0, g := LagrangeReduce(a div N, b, N*c);
  T := Topograph();
  RT := RangeTopograph(a0, b0, c0 : top := T);

  g1 := g[1][1]; g2 := g[1][2]; g3 := g[2][1]; g4 := g[2][2];

  i := 0;
  repeat
    i +:= 1;
    L := AdmissibleVectors(N, g1, g3, i : T := T);
  until not(IsEmpty(L));

  j := 0;
  repeat
    j +:= 1;
    if j gt 1 then
      L cat:= AdmissibleVectors(N, g1, g3, i + j - 1 : T := T);
    end if;
    PropagateValues(RT, i + j);
    C := GetLaxVectors(T, i + j);
    min_L := Minimum([Eval(RT, v[1], v[2]) : v in L]);
    min_C := Minimum([Eval(RT, v[1], v[2]) : v in C]);
  until min_L lt min_C;

  A := [];
  for v in L do
    if Eval(RT, v[1], v[2]) eq min_L then
      h1 := v[1]*g1 + v[2]*g3;
      h2 := N*(v[1]*g2 + v[2]*g4);
      _, u1, u2 := Xgcd(h1, h2);
      a_red, b_red, c_red := UnimodularAction(Matrix([[h1, h2],[-u2, u1]]), a, b, c);
      b_red, c_red := TReduce(a_red, b_red, c_red);
      Append(~A, [a_red, b_red, c_red]);
    end if;
  end for;

  return Sort(A);
end intrinsic;

// NReduce computes the N-reduction of the level N form (a,b,c), completing the implementation of the algorithm in [JR].
intrinsic NReduce(a, b, c, N) -> RngIntElt, RngIntElt, RngIntElt
  {The N-reduction of the level N form (a,b,c)}

  A := NReduceA(a, b, c, N);
  return Explode(A[#A]);
end intrinsic;

// The following functions extend the functionality of NReduce in various ways.

intrinsic NReduceFrickeTwist(a, b, c, N) -> RngIntElt, RngIntElt, RngIntElt
  {The N-reduction of the level N form (c*N, -b, a/N)}

  return NReduce(c*N, -b, a div N, N);
end intrinsic;

intrinsic NReduceImproper(a, b, c, N) -> RngIntElt, RngIntElt, RngIntElt
  {The improper N-reduction of the level N form (a,b,c), i.e., allowing transformation by [[1,0],[0,-1]]}

  A := NReduceA(a, b, c, N);
  for i in [1 .. #A] do
    A[i][2] := Abs(A[i][2]);
  end for;
  A := Sort(A);
  return Explode(A[1]);
end intrinsic;

intrinsic NReducePlus(a, b, c, N) -> RngIntElt, RngIntElt, RngIntElt
  {The plus N-reduction of level N form (a, b, c) (allowing both improper equivalence and Fricke involution)}

  a_red, b_red, c_red := NReduceImproper(a, b, c, N);
  a_red_plus, b_red_plus, c_red_plus := NReduceImproper(c*N, -b, a div N, N);

  if a_red lt a_red_plus or (a_red eq a_red_plus and b_red le b_red_plus) then
    return a_red, b_red, c_red;
  else
    return a_red_plus, b_red_plus, c_red_plus;
  end if;
end intrinsic;

// Returns whether a integral quadratic form is primitive.
function IsPrimitive(a, b, c)
  return GCD([a,b,c]) eq 1;
end function;

// Returns whether a integral quadratic form is positive definite.
function IsPositiveDefinite(a, b, c)
  return (b^2 - 4*a*c lt 0) and a ge 1;
end function;

function IsOneReduced(a, b, c)
  return IsPositiveDefinite(a, b, c)
  //return IsPrimitive(a, b, c) and IsPositiveDefinite(a, b, c)
    and Abs(b) le a
    and a le c
    and (not(Abs(b) eq a or a eq c) or b ge 0);
end function;

function IsPrimitiveOneReduced(a, b, c)
  return IsPrimitive(a, b, c) and IsPositiveDefinite(a, b, c)
    and Abs(b) le a
    and a le c
    and (not(Abs(b) eq a or a eq c) or b ge 0);
end function;

intrinsic LevelOneClasses(D) -> SetEnum
  {Returns a set of reduced representatives for the level 1 classes of discriminant D.}
  assert D lt 0 and (D mod 4 eq 0 or D mod 4 eq 1);

  forms := {};
  bound := Isqrt(-D div 3);
  for a in [1 .. bound] do
    for b in [-a + 1 .. a] do
      is_div, c := IsDivisibleBy(b^2 - D, 4*a);
      if is_div and IsOneReduced(a, b, c) then
        Include(~forms, [a, b, c]);
      end if;
    end for;
  end for;

  return forms;
end intrinsic;

intrinsic LevelOnePrimitiveClasses(D) -> SetEnum
  {Returns a set of representatives for the level 1 primitive classes of discriminant D.}
  assert D lt 0 and (D mod 4 eq 0 or D mod 4 eq 1);

  L := LevelOneClasses(D);
  return [l : l in L | IsPrimitive(l[1], l[2], l[3])];
end intrinsic;

intrinsic LevelNEquivalentForms(a, b, c, N) -> SeqEnum[SeqEnum[RngIntElt]]
  {Return the reduced level N forms which are 1-equivalent to (a,b,c).}

  gamma0 := GammaUpper0(N);
  coset_reps := CosetRepresentatives(gamma0);

  forms := {};
  for rep in coset_reps do
    mat := Matrix(Integers(), 2, 2, Eltseq(rep));
    a_equiv, b_equiv, c_equiv := UnimodularAction(mat, a, b, c);
    if not(IsDivisibleBy(a_equiv, N)) then
      continue;
    end if;
    a_equiv, b_equiv, c_equiv := NReduce(a_equiv, b_equiv, c_equiv, N);
    Include(~forms, [a_equiv, b_equiv, c_equiv]);
  end for;

  return Setseq(forms);
end intrinsic;

intrinsic IsLevelNPrimitive(a, b, c, N) -> BoolElt
  {Returns if a the level N quadratic form (a, b, c) is N-primitive.}

  return Gcd([a div N, b, c]) eq 1;
end intrinsic;

intrinsic LevelNClasses(D, N) -> SeqEnum
  {Returns the reduced level N forms of discriminant D.}

  level_one_forms := LevelOneClasses(D);
  return &cat[LevelNEquivalentForms(l[1], l[2], l[3], N) : l in level_one_forms];
end intrinsic;

intrinsic LevelNPrimitiveClasses(D, N) -> SeqEnum
  {Returns a reduced representative for each primitive class of forms of level N and discriminant D.}

  forms := { PowerSequence(Integers()) | };
  for d in Divisors(N) do
    is_div, D_dsq := IsDivisibleBy(D, d^2);
    if is_div and (D_dsq mod 4 eq 0 or D_dsq mod 4 eq 1) then
      N_d := N div d;
      level_one_forms := LevelOneClasses(D_dsq);
      for one_form in level_one_forms do
        a := d*one_form[1]; b := d*one_form[2]; c := d*one_form[3];
        equiv := LevelNEquivalentForms(a, b, c, N);
        for form in equiv do
          if IsLevelNPrimitive(form[1], form[2], form[3], N) then
            Include(~forms, form);
          end if;
        end for;
      end for;
    end if;
  end for;

  return Setseq(forms);
end intrinsic;

intrinsic ReducedIndices(prec, N) -> SeqEnum
  {Returns the set of level N primtitive reduced representatives for each discriminant up to prec.}
  indices := [];
  for k in [1 .. prec div 4] do
    D := -4*k + 1;
    Append(~indices, <D, LevelNPrimitiveClasses(D, N)>);
    D -:= 1;
    Append(~indices, <D, LevelNPrimitiveClasses(D, N)>);
  end for;

  return indices;
end intrinsic;