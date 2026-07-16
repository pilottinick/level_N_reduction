// Implementation of
// Reduction Theory for Bineary Quadratic Forms with Level
// by Jennifer Johnson-Leung and Brooks Roberts
intrinsic SReduce(a, b, c) -> RngIntElt, RngIntElt, RngIntElt, AlgMatElt
  {Apply [[0,-1],[1,0]] if a > c or if a = c and b < 0}

  if a lt c or (a eq c and b ge 0)then
    return a, b, c, Matrix([[1,0],[0,1]]);
  else
    return c, -b, a, Matrix([[0,-1],[1,0]]);
  end if;
end intrinsic;

intrinsic TReduce(a, b, c) -> RngIntElt, RngIntElt, AlgMatElt
  {Apply a power of [[1,0],[1,1]] so that -a < b <= a without changing a}

  two_a := 2*a;
  m := (a - b) div two_a;
  c +:= m*b + m^2*a;
  b +:= m*two_a;
  return b, c, Matrix([[1,0],[m,1]]);
end intrinsic;

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

function AdmissibleVectors(N, g1, g3, i : T := Topograph())
  K := [];
  for v in GetLaxVectors(T, i) do
    if Gcd(v[1]*g1 + v[2]*g3, N) eq 1 then
      Append(~K, v);
    end if;
  end for;

  return K;
end function;

intrinsic Eval(a, b, c, x, y) -> Assoc
  {Value of the quadratic form (a, b, c) at (x, y)}

  return a*x^2 + b*x*y + c*y^2;
end intrinsic;

intrinsic UnimodularAction(u::AlgMatElt, a::RngIntElt, b::RngIntElt, c::RngIntElt) -> RngIntElt, RngIntElt, RngIntElt
  {The image of the the quadratic form (a, b, c) under the action of the unimodular matrix u = [[g1,g2],[g3,g4]]}

  g1 := u[1][1]; g2 := u[1][2]; g3 := u[2][1]; g4 := u[2][2];
  return Eval(a, b, c, g1, g2), 2*a*g1*g3 + b*(g2*g3 + g1*g4) + 2*c*g2*g4, Eval(a, b, c, g3, g4);
end intrinsic;

intrinsic NReduceA(a, b, c, N) -> SeqEnum
  {Compute the forms which are N-equivalent to the level N form (a,b,c) and such that a is minimal and -a < b <= a.
   The list is sorted in order of increasing b.}

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

intrinsic NReduce(a, b, c, N) -> RngIntElt, RngIntElt, RngIntElt
  {The N-reduction of the level N form (a,b,c)}

  A := NReduceA(a, b, c, N);
  return Explode(A[#A]);
end intrinsic;

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
  {The plus N-reduction of level N form (a, b, c) (allowing improper equivalence and Fricke involution)}

  a_red, b_red, c_red := NReduceImproper(a, b, c, N);
  a_red_plus, b_red_plus, c_red_plus := NReduceImproper(c*N, -b, a div N, N);

  if a_red lt a_red_plus or (a_red eq a_red_plus and b_red le b_red_plus) then
    return a_red, b_red, c_red;
  else
    return a_red_plus, b_red_plus, c_red_plus;
  end if;
end intrinsic;

function IsPrimitivePositiveDefinite(a, b, c)
  return GCD([a,b,c]) eq 1 and b^2 - 4*a*c lt 0 and a ge 1;
end function;

function IsOneReduced(a, b, c)
  return IsPrimitivePositiveDefinite(a, b, c) 
    and Abs(b) le a
    and a le c
    and (not(Abs(b) eq a or a eq c) or b ge 0);
end function;

intrinsic LevelOneClasses(D) -> SetEnum
  {}
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

intrinsic LevelNEquivalentForms(a, b, c, N) -> SetEnum
  {}

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

  return forms;
end intrinsic;

intrinsic IsLevelNPrimitive(a, b, c, N) -> SetEnum
  {}

  return Gcd([a div N, b, c]) eq 1;
end intrinsic;

intrinsic LevelNClasses(D, N) -> SetEnum
  {}

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

  return forms;
end intrinsic;

intrinsic ReducedIndices(prec, N) -> SeqEnum
  {}

  indices := [];
  for k in [1 .. prec div 4] do
    D := -4*k + 1;
    Append(~indices, <D, LevelNClasses(D, N)>);
    D -:= 1;
    Append(~indices, <D, LevelNClasses(D, N)>);
  end for;

  return indices;
end intrinsic;