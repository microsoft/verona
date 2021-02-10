# Programmable Control Flow

New language features:

* Lambda syntax.
* Negation in types to indicate that some type won't be thrown.
* Automatically convert an exception that we don't allow in the function signature to a local return value and type check against the return value.
  * Or enclose all bodies with a try/catch that catches Return[$ResultType] and returns the value
  * May be better to catch each Return[T] exception that's not in the `throws` signature and unwrap it

> TODO: `else if` doesn't actually work, as the `if` produces a value, not a lambda.

## Control Flow

```ts
// Control flow module
class NoLabel {}
class TrueBranch {}
class FalseBranch {}
class Break[L] {}
class Continue[L] {}

class Return[T]
{
  value: T;
}

class return[T = None]
{
  create(x: T = None) throws Return[T]
  {
    throw Return[T](x)
  }
}

class break[L = NoLabel]
{
  create() throws Break[L]
  {
    throw Break[L]
  }
}

class continue[L = NoLabel]
{
  create() throws Continue[L]
  {
    throw Continue[L]
  }
}

if[T](cond: Bool, ontrue: ()->T): T | TrueBranch | FalseBranch
{
  match (cond)
  {
    true =>
    {
      match (ontrue())
      {
        FalseBranch =>
        {
          // If the body returns FalseBranch, intercept it and return
          // TrueBranch instead. This allows a following `else` to distinguish
          // between a branch that wasn't taken and a branch that was taken
          // but itself ended in an `if` with no taken branch.
          TrueBranch
        }

        let v: T => { v }
      }
    }

    false => { FalseBranch }
  }
}

else[T, U](prev: T | TrueBranch | FalseBranch, onfalse: ()->U): T | U
{
  match (prev)
  {
    TrueBranch => { TrueBranch }

    FalseBranch =>
    {
      match (onfalse())
      {
        FalseBranch =>
        {
          // Intercept a FalseBranch result in the same way that `if` does.
          TrueBranch
        }

        let v: U => { v }
      }
    }

    let v: T => { v }
  }
}

while[L = NoLabel, T, U](cond: ()->Bool, body: ()->(T throws U))
  : throws U \ (Break[NoLabel] | Break[L] | Continue[NoLabel] | Continue[L])
{
  match (cond())
  {
    true =>
    {
      try
      {
        body()
        continue
      }
      catch
      {
        Break[NoLabel] | Break[L] => {}
        Continue[NoLabel] | Continue[L] => { while[L] cond body }
      }
    }

    false => {}
  }
}

for[L = NoLabel, T, U, V](iter: Iterator[T], body: T->(U throws V))
  : throws V \ (Break[NoLabel] | Break[L] | Continue[NoLabel] | Continue[L])
{
  while {has_next iter}
  {
    try
    {
      body(iter())
      continue
    }
    catch
    {
      // Propagate Break[NoLabel] to the enclosing while.
      Break[L] => { break }
      Continue[NoLabel] | Continue[L] => { next iter }
    }
  }
}
```

## Boolean Logic

```ts
// Boolean logic module
interface ToBool
{
  bool(self): Bool;
}

// `and` and `or` take lambdas for the rhs to allow short-circuiting, i.e. lazy
// evaluation. However, any type that has an apply method that returns itself
// can be passed as well, allowing strict evaluation.
and[T: ToBool, U](a: T, b: ()->U): T | U
{
  match (bool a)
  {
    // Because `a` is "falsey", return it directly.
    false => { a }
    true => { b() }
  }
}

or[T: ToBool, U](a: T, b: ()->U): T | U
{
  match (bool a)
  {
    // Because `a` is "truthey", return it directly.
    true => { a }
    false => { b() }
  }
}
```

## Example Usage

```ts
// Because `f` can't throw Return[U64], the `return` below becomes a local
// return of U64.
f(iter: Iterator[U64], accum: U64): U64
{
  try
  {
    for iter
    {
      x =>
      accum = accum + x;

      match (x)
      {
        // non-local return, `for` returns 0
        0 => { return 0 }

        // `for` should finish
        1 => { break }

        // `for` should move to the next element
        2 => { continue }
      }

      // local return
      accum = accum + 1
    }

    accum
  }
  catch
  {
    let $0: Return[$ReturnType] => { $0.value }
  }
}

if (a) { e1 } else if (b) { e2 } else { e3 }
// (apply `else`
//  (tuple
//    [
//      (apply `else`
//        (tuple
//          [
//            (apply `if` (tuple [a (lambda e1)]))
//            (apply `if` (tuple [b (lambda e2)]))
//          ]))
//      (lambda e3)
//    ]))

```
