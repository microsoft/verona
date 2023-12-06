# The Verona language: Features and Motivation 

Verona is an object oriented reference capability language with structural
subtyping and bounded polymorphism. Here we summarize each part and try to
motivate the design decisions taken so far. We also pose design questions that
are yet to be answered.

## Structural type system

We use a structural and algebraic type system rather than a nominal one, since
we find it more flexible.

### Traits
Verona is a structural trait based type system, i.e. types can be described
using combination of methods on an object. E.g. we can describe a type
implementing a `toString` method as
```verona
type ToString = {
    toString(self: Self): String
}
```
and a typed describing numbers implementing addition and subtraction with
```verona
type Number = {
    add(self: Self, other: Number): Number
    sub(self: Self, other: Number): Number
}
```
Traits also support default implementations. E.g. we can add a default
implementation of multiplication as


### Conjunction types
We can easily combine traits using conjunctions, so if we e.g. want to describe
something implementing both `ToString` and `Number`, this can be written as
```verona
type ToStringNumber = ToString & Number
```

This gives us a flexible way to define combinations types on the fly, with an
intuitive syntax. E.g. if we want to describe the type signature of a function
`f` printing two numbers and then returning the result of adding them
together, this would be
```verona
f(n1: Number & ToString, n2: Number & ToString) : Number
```

In contrast, in a nominal system like java, we would have to define a new
interface `ToStringNumber` that extends both `Number` and `ToString`:
```java
interface ToStringNumber extends Number, ToString { ... }

class C {
    static Number f(ToStringNumber n1, ToStringNumber n2) { ... }
}
```

In Verona, a trait with multiple methods can be further simplified by breaking
up each method into what we call an "atomic" trait and combining them with a
conjunction. E.g. the definition of `Number` above is simply syntactic sugar for
```verona
type Number = {
    add(self: Self, other: Number): Number
} & {
    sub(self: Self, other: Number): Number
}
```

#### Capabilities and conjunction
In Verona, each concrete type is tagged with a capability. This reference
capability decides what operations are permitted through and with a reference
depending on the viewpoint. E.g. a reference with capability `mut` allows us to
read and mutate the object, while a capability `iso` is unique and opaque to all
read/write operations.

The conjunction type operator plays well together with capabilities, allowing us
to express capability tags as part of the regular typesystem. E.g. given a class
`C` we can express the type of a reference to an object of class type `C` with
reference capability `mut` as
```verona
type CMut = C & mut
```

By extension this means that polymorphism in Verona doesn't need special
handling of capabilities in type parameters. A type `A[T]` paramaterized on `T`
can simply be instantiated with a type like `C & mut`. The method types of `A`
are then able to express further restrictions on `T` that allows us to define
operations for each instantiation.
```verona
type A[T] = {
    withImm(v : T & imm) : A[T] & imm // immutable A[T] created with immutable T
    withMut(v : T & mut) : A[T] & mut // mutable A[T] created with mutable T
}
```

* TODO: compare with
    - objective c: categories
    - ruby: open classes
    - C#: extension methods

### Disjunction types
Verona also includes disjunction types. Assuming two classes `C` and `D` we can
define the new type
```verona
type COrD = C | D
```

Since in Verona there is no inheritance between classes, this will allow us to
check that a pattern match is exhaustive, and furthermore matching on concrete
type will allow us to do implicit type filtering. E.g. imagine a case match like
this
```verona
// C, D, E are classes

f(x : C | D | E, g : (D | E) -> S) : S {
    match x with {
        (y : C) => { // y : C
            ...
        }
        _ => { // x : D | E 
            g x
        }
    }
}
```
We are able to implicitly filter the type of `x` in the default case.

In a language like ML, the corresponding code would look like
```ocaml
type t = C of c
       | D of d
       | E of e


f (x : t) (g : t -> s) =
    match x with
        C y => ...
        _   => g x
```
Note that `g` cannot have a more refined type than `t -> s`.

#### Type discrimination
In ML we might want to construct a type such as
```ocaml
type s = A of int
         B of int
```
A corresponding Verona type might look like
```verona
type S = I32 | I32
```
but it is now impossible to discriminate between the two cases. This is however
easily fixed by defining `S` instead as
```verona
class A { var i : I32 }
class B { var i : I32 }

type S = A | B
```

#### Explicit nullability
Specifically, we can easily encode nullability with a class `None`
```verona
type A
class None {}

f(x : A | None) : S {
    match x with {
        (y: A) => { // y : A
            ...
        }
        _ => { // x : None
            ...
        }
    }
}
```

### Subtyping and inheritance
Verona is an object oriented system with subtyping with classes and traits. In
short, classes are treated nominally, and traits treated structurally.
Furthermore there is no subtype inheritance between classes. This means a class
can be a subtype of a trait, but never a subtype of a class other than itself.
E.g. given the definitions
```verona
class C {
    equal(self: C, other: C): Bool {...}
}

class D {
    equal(self: D, other: D): Bool {...}
}

type Eq = {
    equal(self: Self, other: Self) : Bool
}

```
both `C` and `D` are subtypes of `Eq`, but neither `C` or `D` are subtypes of
the other.

#### Inheritance
There are two forms of what can be classified as inheritance considered in
Verona. One between classes and one between class and trait.

##### Class inheritance
Inheritance between classes would work as you would expect in a language like
Java, with the exception that it does not imply subtyping. As an example,
consider
```verona
class C {
    equal(self: C, other: C): Bool {...}
    f(...) : ... {...}
}

// here we want to reuse functionality of C in D, specifically the method f, but
// we do not get a subtyping relation D <: C
class D : C {
    equal(self: D, other: D): Bool {...}
    g(...): ... { ... f(...) ... }
}
// TODO: find a realistic example
```
If we want to describe a relation between `C` and `D`, we can only do so by
subtyping them under a trait.

* Question: There is a case to be made for not allowing inheritance from a
  class, since this allows unanticipated code reuse. This might however also be
  a positive. The question if we want to allow it remains open.

##### Trait inheritance
Traits can also give us a form of inheritance with their default methods.
```verona
type ToString = {
    toString(s: Self) : String
    print(s: Self) : Unit {
        sys.print(s.toString())
    }
}

class C : ToString {
    toString(s: C) : String {
        ...
    }
}
// here C will inherit the default implementation of print
```
Here, the definition `ToString` contains a default implementation for the print
method.

* Question: Should traits be allowed to define default fields?

###### External trait inheritance
A possibility is to allow externally declared trait inheritance, i.e.
declaration of trait inheritance can be separated from class definition.
```verona
// file 1

// class C definition
class C {
    toString(s: C) : String
    ...
}

// file 2
// declaration that C implements Printable together with its default method
class C : Printable
```


#### Self types
In Verona we explicitly declare a Self type when the corresponding method should
be dynamically dispatched. I.e. in traits, if we declare the first argument to
be of type Self, any time we call this method on a value with this trait type,
it will lead to dynamic dispatch.

In Verona, the Self type represents the concrete type of which the object is
part. I.e. when defining class types, Self mereley acts as an alias for the
instantiation of this class type itself:
```verona
class C {
    m1(self: Self) : Self
    m2(self: C) : C
    // in verona, these two method signatures are equivalent up to method name.
}
```

For trait types Self acts as as a reference to the underlying type. This allows us to write e.g.
```verona
type Collection[T] = {
    add(self: Self, e: T) : Self
}
```
where the `add` method returns something of the concrete type.

* Question: If we have multiple self types, how do we refer to them?
  ```verona
    type A[T] = {
        f(self: Self) : { g(self: Self /* Self1 */) : Self /* Self2 */ }
        // if we want Self1 and Self2 to refer to different types, can we write e.g.
        // A[T].Self?
    }
  ```
  The above example could probably be desugared into
  ```verona
    type A[T] = {
        type Self
        type Anon = {
            type Self
            g(self: A[T].Anon.Self) : A[T].Anon.Self
        }
        f(self: A[T].Self) : A[T].Anon
    }
  ```


## Dynamic and static dispatch
Method calls can be dynamically or statically dispatched based on the type of
the reciever. If the reciever is a concrete class type, the compiler will
produce a static dispatch. If on the other hand, it is more complex, e.g. a
disjunction type or a trait type, the method call will be dispatched
dynamically. The compiler will of course try to optimize if the complex type is
simple enough, e.g. `C | D` in the example below, but this happens on a case by
case basis and the language specification will promise no such thing.
```verona
// classes C and D has method toString
let x : C = C::create()
let s = x.toString() // static dispatch, since C is a concrete (closed) type

let y : C | D = if (...) { C::create() } else { D::create() }
let t = y.toString() // dynamic dispatch, since C | D is a complex (albeit closed) type
// it could however be optimized into something like this, since C | D is a closed world type
let t = match y with {
    C => y.toString() // allows method call inlining since call is now static
    D => y.toString()
}

let z : ToString = if (...) { C::create() } else { D::create() }
let u = z.toString() // dynamic dispatch, since ToString is an open type
```

### Type predicates
Type predicates allows us to describe assumptions about subtyping. These can be
written on both type and method level, and both has their distinct uses.
Specifically, predicates on type level allow us to express F-bounded
polymorphism, while method level predicates are useful in reducing code
duplication, somewhat akin to type classes.

In verona these predicates can be added to types and method signatures using the
keyword `where`. Here are a couple of examples:
```verona
// type Hashable
// type Eq
// type ToString

class HashMap[K, T] where (K <: Hashable & Eq) { // type level where clause
    ...
}

type StringStream = {
    put[T](self: Self, o : T) : Unit where (T <: ToString) // method level where clause
    ...
}
```

#### Type-level `where` clauses
As mentioned above, type predicates on type level are useful when we have to
describe bounds on type parameters. As an example, consider the class of
red-black tree:
```verona
class LT {}
class EQ {}
class GT {}

type Ord = LT | EQ | GT

type Comparable[T] = {
    compare(self: Self, other: T): Ord
}

class RBTree[T] where (T <: Comparable[T]) { ... }
```

The constraint `T <: Comparable[T]` is exactly the constraint that would be
present for the corresponding definition of a red-black tree in a system with
F-bounded polymorphism.

Also consider a polymorphic type describing a graph:
```verona
type GNode[E] = {
    edges(self: Self) : Iterable[E]
}

type GEdge[N] = {
    src(self: Self) : N
    dst(self: Self) : N
}

type Graph[N, E] = (N <: GNode[E]) & (E <: GEdge[N])
```
Note that type constraints such as `N <: GNode[E]` and `E <: GEdge[N]` are part
of the syntax for types, so we can combine them using conjunctions and
disjunctions as in the definition of `Graph[N, E]`.

Continuing we can describe a graph where each node contains a value:
```verona
type Container[V] = {
    val(self: Self) : V
}

type Equal[V] = {
    equals(v1: V, v2: V) : Bool
}

type ValueGraph[V, N, E] = Graph[N, E] & (N < Container[V])
```
Note that we can simply combine the previous definition of `Graph[N, E]` with a
new constraint describing the new constraint on nodes.

We can write a simple search algorithm polymorphic in the type of nodes and edges.
```verona
searchGraph[V, N, E](n : N, v : V) : Option[N]
    where ValueGraph[V, N, E] & (V <: Equal[V]) & (N <: Comparable[N])
{
    let visited = Set[N]::create();
    let pending = Queue[N]::create({n});

    while (!pending.empty()) {
        let next = pending.dequeue();

        if (!visited.contains(next)) {
            visited.add(next);
            if (next.val().equals(v)) {
                Some(next)
            }
            else {
                for (edge : next.edges()) {
                    pending.enqueue(edge.dst());
                }
            }
        }
    }

    None
}
```

#### Method-level `where` clauses
Adding type constraints to methods, we can condition the existance of a method
depending on the specific type parameters. Going back to the red-black tree
example above, we could add a `print` method that will exist only when the
type parameter to `RBTree` implements the `Printable` trait:
```verona
class RBTree[T] where (T <: Comparable[T]) {
    ...
    print(self: Self) : Unit where (T <: Printable) {
        ...
    }
}
```

##### `where` clauses and capabilities
Another use for method-level where clauses is to restrict what methods can be
called depending the capabilities of type parameters.
```verona
class Ref[T] {
    var val : T
    get(self: Self) : T where T <: mut | imm // i.e. not iso
    {
        val
    }
    set(self: Self, v : T) : T { val = v }
}

class Array[T] {
    // uses Ref[T] internally
    ...
    get(self: Self, idx : U32) : T where T <: mut | imm // i.e. not iso
    {
        ...
    }
    set(self: Self, idx : U32, v : T) : T { ... }
}

class HashMap[K, T] {
    ... // built on top of Array[T]
    get(self : Self, k : K) : T where T <: mut | imm // i.e. not iso
    {
        ...
    }
    set(self : Self, k : K, v : T) : T { ... }
}
```
In a system without method-level `where` clauses, we would need to bifurcate each
class on whether the class should be able to handle iso values or not, e.g.
having to define both `IsoRef[T]` and `Ref[T]`. Furthermore, `where` clauses on
methods are very intentional, and allows us to give better error messages. E.g.
"`Ref[iso & A]::get()` not available since `iso & A </: imm | mut`" will
hopefully help the programmer more than "`Ref[iso & A]` is not a valid
instantiation of `Ref[T]`".

##### Default methods and type predicates
Together with type predicates we might be able to split part of `ToString` (see
[Trait Inheritance](#trait-inheritance)) into a `Printable` trait. `Printable`
includes a default implementation in the case where the concrete type also
implements the `ToString` trait:
```verona
type ToString = {
    toString(s: Self) : String
}

type Printable = {
    print(s: Self) : Unit // this will have to be defined if Self </: ToString
    print(s: Self) : Unit where (Self <: ToString) {
        sys.io.out(s.toString())
    }
}

class C : Printable {
    toString(s: C) : String {
        ...
    }
}

class D {
    print(s: D) : Unit {
        ...
    }
}
```
Here both `C` and `D` are subtypes of `Printable`, but only `C` inherits its
default implementation.
