# Object Algebras

To make this statically exhaustive, we need to use closed world types for nodes in the graph instead of open world types.

```ts
using "numbers";
using "boolean";

class NotAlg {}

interface Alg[T, S, R]
{
  (self: Self, alg: Alg[T, T, R], some: S): R;
}

// Integer Expressions
class Lit
{
  x: I32;
}

class Add[T]
{
  left: T;
  right: T;
}

type Exp1 = Lit | Add[Exp1];

class IntEval[T]
{
  (self: Self, alg: Alg[T, T, I32], e: Lit | Add[T]): I32
  {
    match e
    {
      { e: Lit => e.x }
      { e: Add[T] => alg(alg, e.left) + alg(alg, e.right) }
    }
  }
}

// Add a new operation
class IntPrint[T]
{
  (self: Self, alg: Alg[T, T, String], e: Lit | Add[T]): String
  {
    match e
    {
      { e: Lit => e.x.string() }
      { e: Add[T] => alg(alg, e.left) " + " alg(alg, e.right) }
    }
  }
}

// Add new types
class Boolean
{
  x: Bool;
}

class Iff[T]
{
  cond: T;
  ontrue: T;
  onfalse: T;
}

// Add the operations for the new types
class BoolEval[T]
{
  (self: Self, alg: Alg[T, T, Bool], e: Boolean | Iff[T]): Bool
  {
    match e
    {
      { e: Boolean => e.x }
      { e: Iff[T] =>
        if alg(alg, e.cond) { alg(alg, e.ontrue) } else { alg(alg, e.onfalse) }
      }
    }
  }
}

class BoolPrint
{
  (self: Self, alg: Alg[T, T, String], e: Boolean | Iff[T]): String
  {
    match e
    {
      { e: Boolean => e.x.string() }
      { e: Iff[T] =>
        "if " alg(e.cond) " then " alg(e.ontrue) " else " alg(e.onfalse)
      }
    }
  }
}

// Generic combinator.
class Merge[T, S1, S2, R]
{
  left: Alg[T, S1, R];
  right: Alg[T, S2, R];

  (self: Self, alg: Alg[T, T, R], e: S1 | S2): R
  {
    match e
    {
      { e: S1 => left(alg, e) }
      { e: S2 => right(alg, e) }
    }
  }
}

class Seq[A, B]
{
  seq: Array[Alg[A, B]];

  create[T...: Alg[A, B]](from: T...): Seq[A, B]
  {
    let self = new (Array);
    self.seq.pushn(from);
    self
  }

  (self: Self, alg: Alg[A, B], e: A): B | throw NotAlg
  {
    for self.seq.values()
    {
      elem =>
      try
      {
        return elem(alg, e);
      }
      catch
      {
        { NotAlg => }
      }
    }

    throw NotAlg
  }
}

eval(e: Exp): I32 | Bool | throw NotAlg
{
  let alg = Seq(IntEval, BoolEval);
  alg(alg, e)
}

print(e: Exp): String | throw NotAlg
{
  let alg = Seq(IntPrint, BoolPrint);
  alg(alg, e)
}
```

## Pretty Print

```ts

class C1
{
  f1: C2;
  f2: C1 | None;
}

class C2
{
  f1: I32;
  f2: Bool;
}

class PrintC1
{
  (self: Self, alg: Alg[C1 | C2 | None, String], x: C1): String | throw NotAlg
  {
    "C1: " alg(alg, x.f1) ", " alg(alg, x.f2)
  }
}

class PrintC2
{
  (self: Self, alg: Alg[I32 | Bool, String], x: C2): String | throw NotAlg
  {
    "C2: " alg(alg, x.f1) ", " alg(alg, x.f2)
  }
}

class PrintNone
{
  (self: Self, alg: Alg[???, String], x: None): String | throw NotAlg
  {
    "None"
  }
}

class PrintI32
{
  (self: Self, alg: Alg[???, String], x: I32): String | throw NotAlg
  {
    "I32: " x.string()
  }
}

class PrintBool
{
  (self: Self, alg: Alg[???, String], x: Bool): String | throw NotAlg
  {
    "Bool: " x.string()
  }
}

class Seq[]
{
  create(
}

print(x: C1 | C2): String | throw NotAlg
{
  let alg = Seq(PrintC1, PrintC2, PrintNone, PrintI32, PrintBool);
  alg(alg, x)
}

```
