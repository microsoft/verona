# Reachability

To specialise the polymorphism in the IR for lowering we need to perform reachability
analysis.  This determines which classes and methods are being accessed in the program

```
  class Foo[X]
  {
      static bar(): Foo[X] {...}

      map[Y] (f: Fun[X,Y]): Foo[Y]
  }

  main()
  {
      var x = Foo[int].bar()   // (A)
      ...

      // f: Fun[int, double]
      var y = x.map(f);  // (B)
  }
```
Reachability starts with the program entry point, we will assume that is `main`.
For each `Method` and `Class`, we visit we will generate new reachable items that
must be visited transitively. Once, we have visited all reachable items, we know
which specialisations must be codegened.

So for a method, we look at all the types it mentions, and the methods it calls. For `main` above,
for the line marked with `(A)` we will generate,
```
   Foo[int]
   static Foo[int]::bar
```
Then on the line marked with `(B)` we will generate,
```
   Foo[int]::map[double]
```
as this is the instantiation for the class that has been inferred by type inference, and
```
   Foo[double]
```
as this is the inferred return type.  For the method, we will look into their bodies to see what types they use,
and for the classes, we generate a layout for the fields, add the type of each field as reachable, and also add the
`finaliser` as a reachable method.

The situation is slightly more complex for dispatch on an interface type:
```
  interface HasIterator[X]
  {
    getIterator(): Iterator[X]
  }

  f(i: HasIterator[X])
  {
    ...

    i.getIterator();
  }
```
Here reachability will mean that any class that satisfies a `HasIterator` interface, then its `getIterator` method will be considered reachable.

This could be refined by considering, which classes could reach `i`.  For instance, classes that are never allocated, do not need this method, or more precise flow analysis could be applied, or we could track weakening to an interface type.
