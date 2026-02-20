# TODO

* Parsing
  * Lookup
    * Semantics
      - Local first, 
      - use second,
      - if not in either then move up a level.
      - ambiguity if multiple definitions at a level, or multiple definitions brought into scope at a level.

    * Move up a level once all the `use` statements at that level have been resolved. 
    * Look inside all `use` at a level to check for ambiguity.

  * Lookdown
    * `use T1::T2::...::Tn` brings the symbol table in Tn into the current scope.
    * `use T1::T2::...::Tn as Alias` brings the symbol table in Tn into the current scope under the name Alias.
    * Second not needed, that is just `type Alias = T1::..::Tn`
  * Lazy
    * Prevent ambiguous overloading with different lazyness
    * 
  * Int and string literals
  * Syntax for pattern matching
    * Just on type?
  * Warn for operator precedence that is not expected...
  * Lambdas? Converting function into a value
    x map (y => y + 1)
  * Uniform syntax over match and lambda
    (x: T => expr)
    and
    match e
      x: T1 => expr1
      x: T2 => expr2
    * Can we do untyped lambda, and match without the var?
      match b
        True => expr1
        False => expr2
      and
        l map (x => x + 1)
    * Perhaps have a 
       group 
         lambda1
         lambda2
         ...
       This forms a group of lambdas that are dispatched by pattern matching.
       

* Type checking

* Code generation
  * Dictionary passing
  * Pattern match
  * Inline function with lazy arguments

* Reference types?   Leave for later as requires more type checking and elaboration to be nice.
  Function may return either a reference type or not.  No disjunction over ref?
     e.g.  (ref Foo) | Int
  This would allow converting .foo into a function symbol, and that could return a reference type.

  Over loading reference types?  Could we provide our own type that has a left and right operator that
  are implicitly handled.  Does this require dynamic dispatch?


# Question

Infix create syntax
```
  struct (left: T) Foo[T] (right: T)
```

Converting function into value?

Optional arguments?
  * `-` is this unary or binary
    `x * - y`


# Resolution pass

1. Resolve `use` and `type` statements
   * Use relative paths to find the module/struct that contains the term
   ```
    module A
      module B
        pass
    module C
      use A
      type Da = B
    module D
      use C::Da
   ```
   This should resolve to 
   ```
    module A
      module B
        pass
    module C
      use ..::A
      type Da = ..::A::B
    module D
      # Original
      #   use C::Da 
      # Find C
      #   use ..::C::Da
      # Expand Da as alias
      #   use ..::C::..::A::B
      # Normalise path
      use ..::A::B
   ```
2. With generics this gets more interesting.
    ```
    module A[T]
      module B
        type Foo = T
    module C
      use A[Int]
      type Da = B
    module D
      use C::Da
      type F = Foo
    ```
    This should resolve to
    ```
    module A[T]
      module B
        type Foo = ..::T
    module C
      use ..::A[Int]
      type D = ..::A[Int]::B
    module D
      # Original
      #   use C::Da
      # Find C
      #   use ..::C::Da
      # Expand Da as alias
      #   use ..::C::..::A[Int]::B
      # Normalise path
      use ..::A[Int]::B
      # Original
      #   type F = Foo
      # Find Foo in scope
      # Foo is in ..::A[Int]::B
      #   type F = ..::A[Int]::B::Foo
      # Simplify Foo as alias
      #   type F = ..::A[Int]::T
      # Simplify type variable to context
      type F = Int
    ```
  Harder example where the type variable gets collapsed?
    ```
    module A[T]
      module B
        type Foo = T
    module C[U]
      use A[U]
      type Da = B
    module D[V]
      use C[V]::Da
      type F = Foo
    ```
    This should resolve to
    ```
    module A[T]
      module B
        type Foo = ..::T
    module C[U]
      use ..::A[U]
      type Da = ..::A[U]::B
    module D[V]
      # Original
      #   use C[V]::Da
      # Find C
      #   use ..::C[V]::Da
      # Expand Da as alias, apply path to all sub terms inside arguments. 
      #   use ..::C[V]::..::A[..::C[V]::U]::B
      # Normalise path
      #   use ..::A[..::C[V]::U]::B
      # Normalise type variable by lookup
      #   use ..::A[V]::B
      # Original
      #   type F = Foo
      # Find Foo in scope
      # Foo is in ..::A[V]::B
      #   type F = ..::A[V]::B::Foo
      # Simplify Foo as alias
      #   type F = ..::A[V]::B::..::T
      #   type F = ..::A[V]::T
      type F = V
    ```
  An even harder example.
    ```
    module A[T]
      module B
        type Foo = T::F
    module C[U]
      use A[U]
      type Da = B
    module D[V]
      use C[V]::Da
      type F = Foo
    module E
      type G = D[D[Int]]::F
    ```

    ```
    module A[T]
      module B
        type Foo = ..::T::F
    module C[U]
      use ..::A[U]
      type Da = ..::A[U]::B
    module D[V]
      # use ..::C[V]::Da
      # use ..::C[V]::..::A[..::C[V]::U]::B
      # use ..::A[V]::B
      use ..::A[V]::B
      # type F = Foo
      # type F = ..::A[V]::B::Foo
      # type F = ..::A[V]::B::..::T::F
      # type F = ..::A[V]::T::F
      type F = V::F
    module E
      # type G = D[D[Int]]::F
      # type G = ..::D[..::D[Int]]::F
      # type G = ..::D[..::D[Int]]::V::F
      # type G = ..::D[Int]::F
      # type G = ..::D[Int]::V::F
      type G = Int::F 
    ```
  When replacing a type alias
    NPath::TA::Path1
  ->
    NPath::TDef[NPath::Ts]::Path2

  where TA is an alias for TDef[Ts]

  What about if the type alias is generic?  I think this means, we need an additional level of lookup for the type arguments.

  ```
    module A
      module B
        type F[T] = List[T]
    module C
      use A::B
      type G = F[Int]::Values
  ```
  so resolution should do
  ```
    module A
      module B
        type F[T] = List[T]
    module C
      use ..::A::B
      type G = ..::A::B::F[Int]::Values
      type G = List[Int]::Values
  ```
  Substitution has to be done on type arguments to generic aliases.

  I think the algorithm will need to expand each use,
  to see if it evaluates to a type alias, and if any component of the path is a type alias,
  then we need to resolve that first.  And mark that alias as in progress.  If that alias
  results in a cycle then we fail to compile.


  ```
  module C
    struct Foo
      struct Bar

  module A
    use B::T1
    use C
    type T2 = Foo
    type T4 = Bar

  module B
    use A::T2
    use C
    type T1 = Foo
    type T3 = Bar
  ```
  This illustrates the complexity of resolution.

  ```
  module C
    struct Foo
      struct Bar

  module A
    use ..::B::T1
    use ..::C
    type T2 = Foo # This can be resolved without resolving T1
    type T4 = Bar # This needs T1 to be resolved first

  module B
    use ..::A::T2
    use ..::C
    type T1 = Foo # This can be resolved without resolving T2
    type T3 = Bar # This needs T2 to be resolved first
  ```
  which resolves as
  ```
  module C
    struct Foo
      struct Bar

  module A
    use ..::B::T1
    use ..::C
    type T2 = ..::C::Foo # This can be resolved without resolving T1
    type T4 = Bar # This needs T1 to be resolved first

  module B
    use ..::A::T2
    use ..::C
    type T1 = ..::C::Foo # This can be resolved without resolving T2
    type T3 = Bar # This needs T2 to be resolved first
  ```
  then
  ```
  module C
    struct Foo
      struct Bar

  module A
    use ..::B::..::C::Foo
    use ..::C
    type T2 = ..::C::Foo # This can be resolved without resolving T1
    type T4 = Bar # This needs T1 to be resolved first

  module B
    use ..::A::..::C::Foo
    use ..::C
    type T1 = ..::C::Foo # This can be resolved without resolving T2
    type T3 = Bar # This needs T2 to be resolved first
  ```
  then
  ```
  module C
    struct Foo
      struct Bar

  module A
    use ..::B::..::C::Foo
    use ..::C
    type T2 = ..::C::Foo # This can be resolved without resolving T1
    type T4 = ..::B::..::C::Foo::Bar # This needs T1 to be resolved first

  module B
    use ..::A::..::C::Foo
    use ..::C
    type T1 = ..::C::Foo # This can be resolved without resolving T2
    type T3 = ..::B::..::C::Foo::Bar # This needs T2 to be resolved first
  ```


