# Type Lists

`T...` means a type list called `T...`. In a generic, if it is given a constraint, the constraint applies to each type in the list, not the list itself. So `T...: U` means a type list of any length such that each element is a subtype of `U`. The type `()` is an empty type list. Unlike a singleton type, this is not a subtype of the empty interface.

`A...` is a tuple of zero or more types. `(A, B...)` is a tuple of one or more types. That is, the type list is flattened into the tuple. `A...->B` is a function type with zero or more arguments. `A->B...->C` is a function type with one or more arguments. That is, the type list is flattened into the arrow. If `A...` is bound to `(C, D)`, you would get `C->D->B`.

Tuple flattening also happens at the value level. This flattening is type-directed: given `x...: A...`, uses of `x...` are expanded to a sequence of values within the enclosing tuple. For example, given `x: A, y...: B...`, if `B...` is bound to a tuple of length 2, `(x, y...)` is a tuple of length 3, whereas `(x, (y...))` is a tuple of length 2, where the second element is another tuple of length 2. If instead `B...` is bound to a tuple of length 1, then `(x, y...)` and `(x, (y...))` are identical.

When destructuring a type list, variables are bound to single elements until there is only one variable left, which then is bound to the remainder of the type list. This applies when pattern matching as well.

```ts
papply[A, B..., C](f: A->B...->C, x: A): B...->C
{
  { y...: B... => f(x, y...) }
}

curry_one[A, B..., C, R: A->B...->C](f: R~>((A, B...)->C)): R
{
  { x: A => papply(f, x) }
}

uncurry[A..., B](f: A...->B)): (A...)->B
{
  { x...: A... => f x }
}

curry_all[A, B..., C, R: A->B...-->C](f: R~>((A, B...)->C)): R
{
  { x: A =>
    match f
    {
      { g: R~>(A->C) => papply(g, x) }
      { _ => curry_all(papply(f, x) }
    }
  }
}
```

```ts
length[T...](x...: T...): USize
{
  match x...
  {
    { () => 0 }
    { _, tl... => 1 + length(tl...) }
  }
}

class Array[T]
{
  create[U...: T](x: U...): Array[T]
  {
    let self = new ();
    self.resize(length(x));
    self.pushn(x);
    self
  }

  push(self: Array[T] & mut, x: T)
  {
    // append x to self
  }

  pushn[U...: T](self: Array[T] & mut, x...: U...)
  {
    match x...
    {
      { () => }
      { hd, tl... => self.push(hd); self.pushn(tl...) }
    }
  }
}
```
