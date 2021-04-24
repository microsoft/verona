# Object Algebras

```ts
using "numbers";
using "boolean";

class NotAlg {}

interface Alg[A, B]
{
  (self: Self, alg: Alg[A, B], some: A): B | throw NotAlg;
}

// Integer Expressions
interface Exp {}

class Lit
{
  x: I32;
}

class Add
{
  left: Exp;
  right: Exp;
}

class IntEval[T]
{
  (self: Self, alg: Alg[Exp, T | I32], e: Exp): T | I32 | throw NotAlg
  {
    match e
    {
      { e: Lit => e.x }
      { e: Add =>
        let left = alg(alg, e.left);
        let right = alg(alg, e.right);

        match (left, right)
        {
          { x: I32, y: I32 => x + y }
          { throw NotAlg }
        }
      }
      { throw NotAlg }
    }
  }
}

// Add a new operation
class IntPrint
{
  (self: Self, alg: Alg[Exp, String], e: Exp): String | throw NotAlg
  {
    match e
    {
      { e: Lit => e.x.string() }
      { e: Add => alg(e.left) " + " alg(e.right) }
      { throw NotAlg }
    }
  }
}

// Add new types
class Boolean
{
  x: Bool;
}

class Iff
{
  cond: Exp;
  ontrue: Exp;
  onfalse: Exp;
}

// Add the operations for the new types
class BoolEval[T]
{
  (self: Self, alg: Alg[Exp, T], e: Exp): T | Bool | throw NotAlg
  {
    match e
    {
      { e: Boolean => e.x }
      { e: Iff =>
        match alg(e.cond)
        {
          { b: ToBool => if b { alg(e.ontrue) } else { alg(e.onfalse) } }
          { throw NotAlg }
        }
      }
      { throw NotAlg }
    }
  }
}

class BoolPrint
{
  (self: Self, alg: Alg[Exp, String], e: Exp): String | throw NotAlg
  {
    match e
    {
      { e: Boolean => e.x.string() }
      { e: Iff =>
        "if " alg(e.cond) " then " alg(e.ontrue) " else " alg(e.onfalse)
      }
    }
  }
}

// Generic sequential combinator.
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
