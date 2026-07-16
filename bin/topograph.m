declare type Topo;
declare attributes Topo: lax_vecs, lax_bases;

declare type RangeTopo;
declare attributes RangeTopo: top, form, values, values_circ;

intrinsic Print(T::Topo)
  {Print T}

  printf "Conway's topograph";
end intrinsic;

intrinsic Print(T::RangeTopo)
  {Print T}

  printf "Range topograph for the form %o", T`form;
end intrinsic;

intrinsic Topograph() -> Top
  {Topograph}

  T := New(Topo);
  T`lax_bases := AssociativeArray();
  T`lax_vecs := AssociativeArray();

  M := RSpace(Integers(), 2);
  v1 := M.1;
  v2 := M.2;

  T`lax_vecs[1] := [v1,v2];
  T`lax_vecs[2] := [v1+v2,v1-v2];

  T`lax_bases[1] := [[v1,v2]];
  T`lax_bases[2] := [[v1,v1+v2],[v2,v1+v2],[v1,v1-v2],[-v2,v1-v2]];

  return T;
end intrinsic;

intrinsic RangeTopograph(a::RngIntElt, b::RngIntElt, c::RngIntElt : top := Topograph()) -> RangeTopo
  {Range topograph for the form (a, b, c)}

  T := New(RangeTopo);
  T`top := top;
  T`form := [a,b,c];
  T`values := AssociativeArray();

  T`values := AssociativeArray();
  T`values[[1,0]] := a;
  T`values[[0,1]] := c;
  T`values[[1,1]] := a + b + c;
  T`values[[1,-1]] := a - b + c;
  T`values_circ := 2;

  return T;
end intrinsic;

intrinsic GetLaxBases(T::Topo, i::RngIntElt) -> Assoc
  {Lax bases involving the ith circle and lower circle}

  Propagate(T, i);
  return T`lax_bases[i];
end intrinsic;

intrinsic GetLaxVectors(T::Topo, i::RngIntElt) -> Assoc
  {Lax vectors in the ith circle}

  Propagate(T, i);
  return T`lax_vecs[i];
end intrinsic;

intrinsic GetForm(T::RangeTopo) -> SeqEnum
  {Underlying quadratic form}

  return T`form;
end intrinsic;

intrinsic GetValues(T::RangeTopo, i::RngIntElt) -> Assoc
  {Values of the quadratic form at the ith circle}

  PropagateValues(T, i);
  vals := T`values;

  L := [];
  for v in GetLaxVectors(T`top, i) do
    Append(~L, <v, vals[Eltseq(v)]>);
  end for;

  return L;
end intrinsic;

intrinsic Eval(T::RangeTopo, x, y) -> Assoc
  {Value of the quadratic form at (x, y)}

  ex, val := IsDefined(T`values, [x,y]);
  if ex then
    return val;
  else
    form := T`form;
    a := form[1];
    b := form[2];
    c := form[3];
    return a*x^2 + b*x*y + c*y^2;
  end if;
end intrinsic;

intrinsic NormalizeVector(v) -> ModTupRngElt
  {Normalize the vector v}

  if v[1] gt 0 or (v[1] eq 0 and v[2] gt 0) then
    return v;
  else
    return -v;
  end if;
end intrinsic;

intrinsic Propagate(T::Topo, i::RngIntElt)
  {Compute T to the ith circle}

  for j in [#T`lax_vecs + 1 .. i] do
    P := [];
    C := [];
    for b in T`lax_bases[j - 1] do
      new_vec := b[1] + b[2];
      Append(~C, new_vec);
      Append(~P, [b[1], new_vec]);
      Append(~P, [b[2], new_vec]);
    end for;
    T`lax_vecs[j] := C;
    T`lax_bases[j] := P;
  end for;
end intrinsic;

intrinsic PropagateValues(T::RangeTopo, i::RngIntElt)
  {Compute the values of T to the ith circle}
  
  Propagate(T`top, i);

  for j in [T`values_circ + 1 .. i] do
    for b in GetLaxBases(T`top, j - 1) do
      v1 := Eltseq(NormalizeVector(b[1]));
      v2 := Eltseq(NormalizeVector(b[2]));
      v2_m_v1 := Eltseq(NormalizeVector(b[2]-b[1]));
      h1 := T`values[v2_m_v1];
      h2 := T`values[v1] + T`values[v2];
      v3 := Eltseq(b[1] + b[2]);
      // Arithmetic progression rule
      T`values[v3] := 2*h2 - h1;
    end for;
  end for;
  T`values_circ := i;
end intrinsic;