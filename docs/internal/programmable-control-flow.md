# Programmable Control Flow

New language features:

* Lambda parameters can be values, which turns the lambda into a case.
  * If it can be a parameter, it is. Otherwise, it is interpreted as a pattern.
  * The parameter position takes a type that can be the right-hand side in an `==` operator with the pattern value. When the lambda executes, it first checks all such equalities, throwing `NoMatch` if any aren't satisfied.
* Negation in types to indicate that some type won't be thrown.
* Automatically convert an exception that we don't allow in the function signature to a local return value and type check against the return value.
  * Or enclose all bodies with a try/catch that catches Return[$ResultType] and returns the value
  * May be better to catch each Return[T] exception that's not in the signature and unwrap it
* Throw `NoCatch` in a catch clause to backtrack
  * A catch that doesn't execute anything rethrows the exception

## Example Usage

```ts
// Because `f` can't throw Return[U64], the `return` below becomes a local
// return of U64.
f(iter: Iterator[U64], accum: U64): U64
{
  // try
  // {
  for iter
  {
    x =>
    accum = accum + x;

    match x
    {
      // non-local return, `for` returns 0
      { 0 => return 0 }

      // `for` should finish
      { 1 => break }

      // `for` should move to the next element
      { 2 => continue }
    }

    // local return
    accum = accum + 1
  }

  accum
  // }
  // catch
  // {
  //   { $0: Return[$ReturnType] => $0.value }
  // }
}

if (a) { e1 } else case {b} { e2 } else { e3 }
// (apply `else`
//  (tuple
//    [
//      (apply `else`
//        (tuple
//          [
//            (apply `if` (tuple [a (lambda e1)]))
//            (apply `case` (tuple [(lambda b) (lambda e2)]))
//          ]))
//      (lambda e3)
//    ]))

```

## Cases

A case can can backtrack by throwing `NoMatch`.
Labelled `NoMatch` gets multi-level backtrack.
A match that doesn't match anything throws `NoMatch`.
`match` can't be written in the source language because we want to detect exhaustive matches.

> TODO: destructure in patterns
> TODO: function level pattern matching?
> TODO: labelled match?

```ts

class Even[T: Number]
{
  ==(self, x: T): Bool
  {
    (x mod 2) == 0
  }
}

class Odd[T: Number]
{
  ==(self, x: T): Bool
  {
    (x mod 2) == 1
  }
}

match x
{
  // { $0: $typeof x => requires(Even[$typeof x] == $0); ... }
  { (Even) => ... }
  // { $0: $typeof x => requires(Odd[$typeof x] == $0); ... }
  { (Odd) => ... }
  // { $0: $typeof x =>
  //   requires((Even[$typeof x] | Odd[$typeof x]) == $0); ...
  // }
  { (Even | Odd) => ... }
  { ... }
}

match x
{
  { true, z: String => ... }
  { false, z: U32 => ... }
  { ... }
}

class C
{
  f(true, x: U32): String { ... }
  // ->
  f$0($1: Bool, x: U32): String | throw NoMatch
  {
    requires (true == $1);
    ...
  }

  f(false, x: F32): String { ... }
  // ->
  f$2($3: Bool, x: F32): String | throw NoMatch
  {
    requires (false == $3);
    ...
  }

  // ->
  f[T: (Bool, U32) | (Bool, F32)](x: T): String | throw NoMatch
  {
    match x
    {
      { $4: Bool, $5: U32 => f$0($4, $5) }
      { $6: Bool, $7: F32 => f$2($6, $7) }
    }
  }
}

```
