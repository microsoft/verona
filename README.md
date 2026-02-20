# InfixLang

This is an experiment in language design.  The aim is explore two things

* Using only union types and dispatch on unions instead of either pattern matching or dynamic dispatch.
* Using an early parse phase to determine the arity and infix nature of functions.

## Building and Testing

```bash
# Configure and build
cmake -S . -B build -G Ninja
cd build
ninja

# Run all tests (compares output against golden files in testsuite/generated/)
ctest

# Update golden files after making changes
ninja update-dump
```

### Test workflow

1. Create a new `.infix` file under `testsuite/` (e.g. `testsuite/parsing/binding/mytest.infix`).
2. Run `ninja update-dump` in the build directory. This both builds and runs `infixlang` on every test, writing golden output files directly into `testsuite/generated/`. There is no need to copy files manually — `update-dump` places them in the correct location.
3. Run `ctest` to verify all tests pass by comparing current output against the golden files.
4. Commit both the `.infix` source file and its generated golden directory.

### Test structure

- **Test discovery**: All `*.infix` files under `testsuite/` are found automatically.
- **Golden output**: For each test `testsuite/path/to/name.infix`, golden files live in `testsuite/generated/path/to/name/` and include per-pass `.trieste` dumps, `exit_code.txt`, `stdout.txt`, and `stderr.txt`.
- **Exit codes**: `0` = success, `1` = expected errors (resolution failures, cycles, etc.).

## VS Code Extensions

This project includes VS Code extensions for syntax highlighting:

- **Infix Language Support**: Syntax highlighting for `.infix` and `.ifx` files
- **Trieste Syntax Highlighting**: Syntax highlighting for `.trieste` AST files

The extensions are automatically available in this workspace. For other workspaces, copy the extension directories from `vscode-extensions/` to `.vscode/extensions/` and reload VS Code.

## Syntax

The syntax is white space sensitive and uses indentation to denote grouping.


## Types

### Primitive Types

We have a few primitive types ... `Bool`, `Int` and `String` for now.

### Union Types

The language is built around union types.  These are arbitrary union types that can be used to combine any existing types.

We write `T | U` to denote a union type of `T` and `U`.  For example, we can define a union type of `Int` and `String` as follows:


### Type Aliases

Type aliases are used to give a name to an existing type.  This is useful for making code more readable and for defining complex types.

```
type IntOrString = Int | String
```

Something of type `IntOrString` can be either an `Int` or a `String`.  This is useful for defining functions that can accept multiple types of arguments.


### Plain old data

The key representation of data is a `struct`.  This is simply a collection of fields, each with a name and a type.  The type can be any existing type, including other structs or union types.

```
struct Person
  name: String
  age: Int
```

This defines a Person type with two fields: `name` of type `String` and `age` of type `Int`.
The indentation indicates that `name` and `age` are fields of the `Person` struct.

This can also be written as a single line:
```
struct Person (name: String) (age: Int)
```

The `struct` can be constructed using the type name followed by the fields in parentheses:
```
  person = Person "Alice" 30
```

We can access the fields of a struct using the dot notation:
```
  person.name
  person.age
```

The `struct` can also be mutated by assigning a new value to a field:
```
  person.age = 31
```

The `struct` can also be generic.  For example, we can define a pair that can hold any type of value as its two components as follows:

```
struct Pair[T, U]
  fst: T
  snd: U 
```

By combining the features covered so far we can define error wrappers:
```
struct Result[T]
  value: T
struct Error
  message: String
type ResultOrError[T] = Result[T] | Error
```

### Recursive types

The `struct` can be recursive, meaning that it can contain a field of its own type.  For example, we can define a `Tree` struct as follows:

```
struct EmptyTree
struct TreeNode[T]
  value: T
  left: Tree[T]
  right: Tree[T]
type Tree[T] = TreeNode[T] | EmptyTree
```

Note we need to use a union type of `Tree` to ensure that the structure can be constructed.

The reader should note that we do not need a different type of `EmptyTree` for each `T`.
This is different from the typical approach in ML or an OO language where each type of empty is distinct: E.g.

```ML
datatype Tree[T] = EmptyTree | TreeNode of T * Tree[T] * Tree[T]
```
or in Java
```Java
class EmptyTree<T> implements Tree<T> { ... }
class TreeNode<T> implements Tree<T> { T value; Tree<T> left; Tree<T> right; ... }
```


## Functions

The language supports functions using the `function` keyword.
Functions can be defined with a name, parameters, and a return type.  The parameters can be of any type, including union types.

```
function fst (a: Int) (b: Int) : Int = a
```

This function takes two `Int` parameters and returns the first one.
The `(a: Int)` specifies that the first parameter is can be referred to as `a` in the body of the function,
and it is of type `Int`.
The `(b: Int)` specifies the second parameter, which is also of type `Int`.
The `: Int` specifies that the return type of the function is `Int`.
The `=` indicates the rest of this block represents the body of the function, which is simply `a` in this case.

The function syntax can also be split across multiple lines for readability:

```
function fst (a: Int) (b: Int) : Int =
  a

function
  fst 
    (a: Int)
    (b: Int)
  : Int =
    a

function
  fst 
    a: Int
    b: Int
  : Int =
    a
```
TODO explain

Function can also be generic, meaning that they can accept parameters of type.  For example, we can define a function that takes two arguments of any type and returns the first as:

```
function[A, B] kay (a: A) (b: B) : A = a
```

### Function with infix notation

Functions can be declared as infix by specify arguments before the function name, e.g.:

```
function (a: Int) + (b: Int) : Int = ...
```

This declares a function symbol `+` that has one left and one right argument.

We can have multiple left and right parameters:
```
function (l1: T1) (l2: T2) f (r1: R1) (r2: R2) (r3: R3): R = ...
```

This declares a function with 2 left parameters and three right parameters.

The arity of a function is defined as a pair of two natural numbers, the first for the number of left parameters, and the second for the number of right parameters. You may not declare the same function multiple times with different arities.

The set of strings that are valid infix, prefix and postfix function names are equal.  That is we can locally define `++` as either an infix, prefix or postfix.


In the following we will assume `+` and `++` are integer addition and string concatenation, respectively.

This also allows us to define functions that are prefix or postfix operators.
For example,
```
function (a: Int) toString : String = int_to_string a
```
defines a postfix operator `toString` that converts an integer to a string.

### Function with union types

One the key features of the language is that there can be multiple functions with the same name, and the compiler will determine which function to call based on the types of the arguments.  This may even involve dynamic case analysis of the arguments at runtime.

Consider our original type of `IntOrString`, we could define two functions to convert this to a string:
```
function (a: Int) toString : String = int_to_string a
function (s: String) toString : String = s
```
with these functions in scope, we actually get an additional function of type
```
function f (a: Int | String) : String = a toString
```
In this function, the call to `toString` will determine, which function to call using the dynamic type of the argument, a.

The implicit function can also be used recursively.  For example, let us take our earlier definition of a tree structure.
We can define a function that counts the number of nodes in a tree as follows:

```
function count (tree: EmptyTree) : Int = 0
function[T] count (tree: TreeNode[T]) : Int =
  1 + (count tree.left) + (count tree.right)
```

This function has two definitions: one for `EmptyTree` and one for `TreeNode[T]`.
The compiler will determine which function to call based on the type of the argument passed to the function.

Inside the body `1 + (count tree.left) + (count tree.right)` is a recursive call to the `count` function,
which will again dispatch to the appropriate definition based on the dynamic type of `tree.left` and `tree.right`.

This combination of union types and function overloading allows for powerful and flexible code that can handle a wide variety of cases without needing explicit type checks or pattern matching.


### Functions with constraints

Functions can be conditionally defined on the existence of other functions.
For instance, we can define a `toString` function on the previous tree type in the following way.

The first case is simple 
```
function (e: EmptyTree) toString : String = "[]"
```
We provide a string that represents the empty tree.

The second case requires the `toString` function to be defined for the type `T`, so that we can print the nodes of the tree:
```
function[T] (e: TreeNode[T]) toString : String where toString: T -> String =
  (e.value toString) ++ "[" ++ (e.left toString) ++ ", " ++ (e.right toString) ++ "]"
```

This is like type-classes in that we require the caller to provide a witness for the
`toString` function for the specific type `T` and returns a `String`.
The syntax for types is a little different to other languages as the arguments can be provided from both the left and right.
Hence, with this we can call `toString` on a `Tree[Tree[IntOrString]]`.

We use `S` as a function signature:
```
  S ::=  T -> S | R
  R ::=  R <- T | T
```
For example,
```
Bool -> Bool <- Int
```
is a signature of a function that takes two arguments, a boolean from the left, and an integer from the right, and returns a boolean.
With this syntax the standard function arrows are backwards to languages like ML.


We can define equality for the tree structure as follows:
```
type eq[T] = (==: T -> Bool <- T) & (!=: T -> Bool <- T)

function[T] (a: TreeNode[T]) == (b: TreeNode[T]) : Bool where eq[T] =
  (a.value == b.value) && (a.left == b.left) && (a.right == b.right)
function[T] (a: TreeNode[T]) == (b: EmptyTree) : Bool =
  false
function[T] (a: EmptyTree) == (b: TreeNode[T]) : Bool =
  false
function (a: EmptyTree) == (b: EmptyTree) : Bool =
  true
```

TODO: Default functions for `false` cases?
```
type different[T,U] = concrete[T] & concrete[U] & notEqual[T, U]
default function[T,U] (a: T) == (b: U) : Bool where different[T,U] = false
default function[T] (a: T) !=  (b: T): Bool where ==: T -> Bool <- T = not (a == b) 
```
Here `concrete[T]` is a built-in type predicate that checks if a type represents a single struct,
and `notEqual[T,U]` is another built-in type predicate that checks if two types are different.



## Aside: Expression problem.

Importantly, this combination of features can be used to solve the classic expression problem, 
where we want to define a function that can handle different types of expressions without needing to modify the function itself.

Consider a simple syntax of mathematical formula
```
struct Add[Expr]
  left: Expr
  right: Expr
struct Sub[Expr]
  left: Expr
  right: Expr
struct IntLiteral
  value: Int
```

This defines three types of expressions: `Add`, `Sub`, and `IntLiteral`.
Importantly, `Add` and `Sub` are generic types that can hold any type of expression as their left and right operands.
We can define a "closed" language of expressions by defining a union type that includes all of these expression types:

```
type Expr = Add[Expr] | Sub[Expr] | IntLiteral
```

We can define functions that operate on each individual expression type using the overloading of functions:
```
function eval (expr: IntLiteral) : Int = expr.value
function[E] eval (expr: Add[E]) : Int where eval: Int <- E =
  eval(expr.left) + eval(expr.right)
function[E] eval (expr: Sub[E]) : Int where eval: Int <- E =
  eval(expr.left) - eval(expr.right)
```

Importantly, the functions `eval` for `Add` and `Sub` have a constraint that the `eval` function must be defined for the type of expression they are operating on.
This allows us to ensure that we can evaluate the left and right operands of the expression.

At this point, the `eval` function can be called with any expression of type `Expr`, and it will dispatch to the appropriate function based on the dynamic type of the expression.

We can then extend the language by adding new expression types without needing to modify the `eval` function for the existing expression types.
For example, we can add a `Mul` expression type as follows:

```
struct Mul[Expr]
  left: Expr
  right: Expr
type Expr2 = Add[Expr2] | Sub[Expr2] | Mul[Expr2] | Int
```

We can then define a new `eval` function for `Mul`:

```
function eval[E] (expr: Mul[E]) : Int where eval: Int <- E =
  eval(expr.left) * eval(expr.right)
```

This allows us to extend the language with new expression types without needing to modify the existing `eval` function for `Add` and `Sub`.

Similarly, we can perform `toString` behaviour for the expression types:

```
function (e: IntLiteral) toString : String = itoa(e.value)

function[E] (e: Add[E]) toString : String where toString: E -> String =
  (e.left toString) ++ " + " ++ (e.right toString)

function[E] (e: Sub[E]) toString : String where toString: E -> String =
  (e.left toString) ++ " - " ++ (e.right toString)

function[E] (e: Mul[E]) toString : String where toString: E -> String =
  (e.left toString) ++ " * " ++ (e.right toString)
```

This provides a `toString` for both `Expr` and `Expr2` types.


##  Modularity

[This is like ML modules]

The language defined so far provides no way to hide implementation details or to group related functions and types together.
To address this, we introduce the concept of modules.  A module is a collection of types, functions, and other modules that can be imported and used in other modules.

A module is defined using the `module` keyword, followed by the name of the module and a block of code that defines the contents of the module.

```
module Math
  type Point

  function create (x: Int) (y: Int) : Point =
    Point x y

  function add (p1: Point) (p2: Point) : Point =
    Point (p1.x + p2.x) (p1.y + p2.y)

  function distance (p1: Point) (p2: Point) : Float =
    let p_diff = add p2 (neg p1)
    sqrt((p_diff.x^2) + (p_diff.y^2))
private
  struct Point
    x: Int
    y: Int

  function neg (p: Point) : Point =
    Point (-p.x) (-p.y)
```

The `private` keyword indicates that the definition of the `Point` struct is not visible outside of the `Math` module.
The module also defines a private function `neg` that negates a `Point`, which cannot be accessed outside of the module.

We can also define type aliases within a module, and make those private as well:

```
type Order[V] = (V `<` V : Int) & (V `>` V : Int) 

module Set[V] where Order[V]
  type T
  function create (value: V) : T
  function insert (tree: T) (value: V) : T
  function count (tree: T) : Int
private
  struct EmptyTree
  struct TreeNode
    value: V
    left: T
    right: T
  type T = EmptyTree | TreeNode

  function count (tree: EmptyTree) : Int = 0
  function count[T] (tree: TreeNode) : Int =
    1 + count(tree.left) + count(tree.right)

  function create (value: V) : T =
    TreeNode value EmptyTree EmptyTree

  function insert (tree: EmptyTree) (value: V) : T =
    create value

  function insert (tree: TreeNode) (value: V) : T =
    if value < tree.value then
      TreeNode tree.value (insert tree.left value) tree.right
    else if value > tree.value then
      TreeNode tree.value tree.left (insert tree.right value)
    else
      tree  // value already exists, do nothing  
```


## Examples

### Abstraction modules example

Consider a simple GUI toolkit

[Idea use generics to provide an extensible GUI toolkit]
[Aim allow button to have a label that is provided by the user GUI component.]

```
module GUI
  type Button[L]
  type Label
  type TextBox

  type Widget[W] = Button[W] | Label | TextBox | W

  type Showable[W] = `get_size W : Box` & `draw W : Unit`

  assert Showable[W] => Showable[Widget[W]]

  function createButton[L] (label: L) : Button
  function createLabel (text: String) : Label
  function createTextBox (placeholder: String) : TextBox

  function setText (widget: Widget) (text: String)
  function getText (widget: Widget) : String

```


## Implementation

### Algorithm for infix, prefix, postfix.

We write [n,"f",m] for a function `f` which consumes `n` arguments before it, and `m` after it.
Note that both `n` and `m` can be zero.

Consider a language

```
E ::= [n, "f", m]
   |  t
   |  (E ... E)

t ::= identifier
     |  literal
     | "f" @ {t,...,t}
```

We use intermediate terms

```
C ::= [n, "f", m] @ {t, ..., t}
    | t

Cs ::= C
     | Cs C
```



```
-----------------------------------------------
Cs :: t, [n + 1, "f", m] @ {ts} :: Es -> 
  Cs, [n, "f", m] @ {ts, t} :: Es


-----------------------------------------------
Cs, [0, "f", m] @ {ts} :: Es ->
  Cs :: [0, "f", m] @ {ts},  Es

-----------------------------------------------
Cs :: [0, "f", n+1] @ {ts}, t :: Es ->
  Cs, [0, "f", n] @ {ts, t} :: Es

-----------------------------------------------
Cs :: [0, "f", 0] @ {ts}, Es ->
  Cs :: "f" @ {ts}, Es

-----------------------------------------------
Cs :: [0, "f1", n+1] @ {ts}, [n + 1, "f2", m] @ {ts} :: Es ->
  Error "f1" requires another "n+1" arguments, but is being called with an `infix` operator "f2" that requires at least one additional a left hand operator.
```

### Dispatch on union types

Introduce a new function for the dispatch.
Internally uses pattern matching.
Large pattern matches compiled to a vtable entry.

### Compiling `where` constraints

These become dictionary passing for the set of required constraints.


## DUMP

We can also define functions with constraints that require certain other functions to be defined for this function to exist.
This is a classic typeclass like approach:

```
struct EmptyList
struct ListNode[T]
  value: T
  next: List[T]
type List[T] = EmptyList | ListNode[T]

function isEmpty[T] (list: EmptyList) : Bool = true
function isEmpty[T] (list: ListNode[T]) : Bool = false

function toString[T] (list: List[T]) : String where `toString T : String` =
  function inner (list: EmptyList) : String = ""
  function inner (list: ListNode[T]) : String  =
    toString list.value ++
      if isEmpty list.next then 
        ""
      else 
        "," ++ inner list.next

  "[" ++ inner list ++ "]"
```



### READING LIST

A calculus for overloaded functions with subtyping.
 Beppe

Gradual Typing with Union and Intersection Types
 Beppe

Design of the Programming Language Forsythe
 John C. Reynolds

Elaborating intersection and union types
  Jana Dunfield


Union Types with Disjoint Switches
  Bruno et al


Pattern Calculus: Computing with Functions and Structures
  Barry Jay

Type checking higher-order polymorphic multi-methods

Polymorphic symmetric multiple dispatch with variance

Finally Tagless Partially Evaluated
  Oleg Kiselyov

## Comments from Mads

Unions of function types?  Can you dispatch/discriminate on this?

Multiple-dispatch parameters?  What happens if there are gaps?
  f (x: A | B)  (y : A)
  f (x: B)  (y : B)
This can't handle {(A,A), (B,A), (B,B)}, but not (A, B).  Is that okay?
Answer:  creating combined functions is demand driven. If someone calls where it could be 
(A,B), then there is a problem.  But if they don't have that then it is fine.



Power of the dot for discoverability?  No answer on the call
After call:
```
function[P, Q] (l: List[P]) map (f: Fun[P, Q]) = 
  function inner (l: EmptyList) = EmptyList
  function inner (l: ListNode[P]) = 
    ListNode (f @ l.value) (inner l.next)
  inner l
```
Then when you have something of type list, then applying an infix map will be an option.

```
l map f map g map h toString
```

This could also head down the postfix operators more
```
function (i: Int) toString = ...

function (l : EmptyList) isEmpty = True
function[T] (l: ListNode[T]) isEmpty = False

function[T] (l: List[T]) toString =
  function inner (l: EmptyList) = ""
  function inner (l: ListNode[T]) =
    l.value toString ++
      if (l.next isEmpty) then
        ""
      else
        "," ++ inner l.next

  "[" ++ inner l ++ "]"
```

Going heavy on the postfix might give the feeling of OO?

## Open Thoughts

How do int literals decide on what they inhabit?

```
  type PositiveIntLiteral = ...
  type IntLiteral = SmallIntLiteral | LargeIntLiteral
  type u64 = u64machine | PositiveIntLiteral
  type s64 = s64machine | IntLiteral

  type float = floatmachine | IntLiteral | FloatLiteral


  function s64ToMachine (s: s64machine) : s64machine = s
  function s64ToMachine (s: PositiveIntLiteral) : s64machine = ...
```
Then we will need to litter around the coercions?

Alternatively, add implicit coercions to the language?

```
  implicit (s: IntLiteral) : u64 = ...
```

Trying to make type inference forward only

Only convert IntLiteral into a numeric type when it flows into the first function that uses it.

Something like:
```
  var i = 0
  i = i + 1
```
would lead to an error like:
```
Cannot select implementation of `+` arguments are `IntLiteral`,
provide a type coercion to specify representation.
```
So we would ideally write
```
  var i: u64 = 0
  i = i + 1
```
which would become
```
  var i: u64 = implicit[IntLiteral, u64](0)
  i = i + implicit[IntLiteral, u64](1)
```

Do similar for strings and other literal types.


## Comments from Sylvan

Can you actually resolve enough in advance to make this work?

Defaults and constructing template parameters:

```
   function[T] f : T = T()
```

This would require a default constructor for T, and also some way to constrain T?

Is there a more sensible example?


What about default parameters for functions?

```
  range (start: Int) (end: Int) with (step: Int = 1)
```

The parser cannot handle this at the moment.

We could add syntax for providing named parameters to functions to override defaults?

```
  range 0 10 with step=2
``` 

Sylvan's example was something like
```
function[T] range (start: T) (end: T) with (step: T = T 1)
```
We could do something like
```
  function[T] range (start: T) (end: T) with (step: T = T 1) where (+: T -> T <- T) & (T: T <- u32) = 
    function inner (current: T) =
      if current >= end then
        EmptyList
      else
        ListNode current (inner (current + step))
    inner start
```
Here `(T: T <- u32)` is a create syntax for a type. 

We only need the `where` on the default for the construction from `1`?

```
  function[T] range (start: T) (end: T) with (step: T = T 1 where T: T <- u32) where (+: T -> T <- T) = 
    function inner (current: T) =
      if current >= end then
        EmptyList
      else
        ListNode current (inner (current + step))
    inner start
```


Could also do
```
  function[T] range (start: T) (end: T) with (step: T = 1 where T == u32) where (+: T -> T <- T) = 
    function inner (current: T) =
      if current >= end then
        EmptyList
      else
        ListNode current (inner (current + step))
    inner start
```

Use would be something like
```
  range 0 10
  range 0.0 1.0 with step=0.1
```



Path dependent types?

Examples
  Iterators in C++
  Red-black tree in snmalloc
  Configuration for snmalloc


Perhaps need a constraint
```
   T::Values : type
```
To specify that a type has an associated type?  This could be implied by the presence of this associated type in a where clause?


What would an iterator look like?

```
type Iterable[C] =
  (current: C::Iterator -> Option[C::Values]) &
  (next: C::Iterator -> C::Iterator) &
  (iterator: C -> C::Iterator)
```

```
function[C, Acc]
    (collection: C)
  fold 
    (f: Fun[(C::Values, Acc), Acc])
  with
    (init: Acc = Acc::default where Acc::default: Acc)
  : Acc 
  where Iterable[C] =
  function inner (it: C::Iterator) (acc: Acc) : Acc =
    match it current with
      None => acc
      v: Some[C::Values] => 
        inner (it next) (f @ (v, acc))
  inner (collection.iterator) init
```

```
struct List[T]
  type Values = T
  type Iterator
  function default : List[T] = <empty array>
  function iterator (list: List[T]) : Iterator = 
    List[T]::Iterator (ref list.head)
  
  function (it: Iterator) current : Option[List[T]::Values] =
    match it.current 
      ln: ListNode => Some ln.value
      el: EmptyList => None

  function next (it: Iterator) : Iterator=
    let current = load it.current
    match current 
      ln: ListNode => it.current = ref ln.next
      el: EmptyList => 

  function insert (it: Iterator) (value: T) =
    let new_node = ListNode value it.current.next
    it.current.next = new_node

  function delete (it: Iterator) =
    let current = it.current
    match current 
      ln: ListNode => store it.current ln.next # TODO ref cases are hard.
      el: EmptyList =>
private
  struct EmptyList
  struct ListNode
    value: T
    next: L
  type L = EmptyList | ListNode

  head: L
  struct Iterator
    current: ref[L]
```

Then we can do
```
  coll fold ((v, acc) -> v + acc) with init=0
```



#### References built in type

```
struct ref[T]
  intrinsic

function load (r: ref[T]) : T = intrinsic

function store (r: ref[T]) (v: T) : T = intrinsic

```

Oh, can we do ML syntax for deref and assign?

```
function ! (r: ref[T]) : T = load r
function (r: ref[T]) := (v: T) : T = store r v
```
Not sure parsers can do `:=` I think `:` is reserved for type annotations.

Some for of reference type class
```
type Reference[R] = (load : R::Values <- R) & (store : R::Values <- R <- R::Values)
```



### Iterator fusion

The following could be optimised into a single pass.
```
x map f map g filter p map h filter q toList
```
The single pass would look something like

```
function[T,U] lift (f: Fun[T,U]) : Fun[T, Option[U]] = (x => Some (f @ x))
function[T] lift_pred (p: Fun[T, Bool]) : Fun[T, Option[T]] = (x => if (p @ x) then Some x else None)

function[T, U] (o: None) >> (f: Fun[T, Option[U]]) : Option[U] = None
function[T, U] (o: Some[T]) >> (f: Fun[T, Option[U]]) : Option[U] = f @ o.value

function[T,S,U] compose (f: Fun[S, Option[T]]) (g: Fun[T, Option[U]]) : Fun[S, Option[U]]
  = x => f @ x >> g

module IterTools
  type IteratorTrait[I] = (next: I -> Bool) & (current: I -> I::Values) & (I::Values :  type)

  type Iterator[I,U]
  
  function (iter: Iterator[I,U]) next: Bool =
    if (iter.current next)
      match (iter.partialFunction @ (iter.current current)) 
        None => iter next
        Some v => true
    else
      false

  function (iter: Iterator[I,U]) current: U =
    (iter.partialFunction @ (iter.current current)) 
    else
      error "Iterator in invalid state"

  function[I] (iter: I) exists (p: Fun[I::Values, Bool]) : Bool where IteratorTrait[I] =
    if iter next then
      let c = iter current
      if p @ c then
        Some c
      else
        iter exists p
    else
      None

  function[I,U] (i: I) map_filter (f: Fun[I::Values, Option[U]]) : Iterator[I, U] =
    Iterator i f

  function[I,T,U] (i: Iterator[I,T]) map (f: Fun[T, U]) : Iterator[I,U] =
    Iterator i.current (lift f >> i.partialFunction)

  function[I,T] (i: Iterator[I,T]) filter (p: Fun[T, Bool]) : Iterator[I,T] =
    Iterator i.current (lift_pred p >> i.partialFunction)

  ## Need to specify this doesn't overlap with the concrete cases above?
  ## Uses != in the type is this crazy
  ## Alternative use a notion of default.  This means other packages can provide there own notion of map and filter.
  ## but get these if they don't
  function[I,T,U] (i: I) map (f: Fun[T,U]): Iterator[I,U] where IteratorTrait[I] & (I != Iterator[_,_]) =
    i map_filter (lift f)

  function[I,T] (i: I) filter (p: Fun[T,Bool]) : Iterator[I,T] where IteratorTrait[I] & (I != Iterator[_,_]) =
    i map_filter (lift_pred p)

private
  struct Iterator[I,U] where IteratorTrait[I]
    type Values = U
    current: I
    partialFunction: Fun[I::Values, Option[U]]



struct RangeIterator
  current: Int
  last: Int
  step: Int
  type Values = Int

function (ri: RangeIterator) next : Bool =
  if ri.current + step > ri.last then
    false
  else
    ri.current = ri.current + step
    true

function (ri: RangeIterator) current : Int = ri.current

function (start: Int) to (end: Int) with (step: Int = 1) : RangeIterator =
  RangeIterator (start - step) end step
```




## Refinement types?

Could we use expression of type bool as refinement types for matching?

```
function (a: Int { _ == 0 }) ! : Int = 1
function (a: Int { _ > 0 }) ! : Int = (a-1)! * a
```

This feels like a super advanced feature that could do nice pattern matching.



## Defaults

We can provide default implementations for functions that can be overridden by more specific implementations.
This leads to compact definitions for some functions like `and` and `or`.
```
default function (a: Bool) or (b: Bool): Bool = True
function (a: False) or (b: False): Bool = False

default function (a: Bool) and (b: Bool): Bool = False
function (a: True) and (b: True): Bool = True
```


## Versioned files

File should start with a version number for the language.

If there isn't one, then it is assumed to be the current and a warning is issued.

```
use v1.0
```

This will allow for upgrading files to new versions of the language.
We want transparent tools that can propose fixes if the semantics would change, so that
previous semantics can be preserved if desired.



## Monad

Can we use the associate types to handle monads?
Seems is would require higher kinded types?

```
module OptionMonad
  type M[T] = None | Some[T]

  function (x: None) >> (f: Fun[T, M[S]]) : None = None
  function (x: Some[T]) >> (f: Fun[T, M[S]]) : M[S] = f @ x.value

  function (v: T) lift : M[T] = Some v


   >> : M[T] -> M[S] <- Fun[T, M[S]]
```


```
module MonadFold
  List[T] ->   <- Fun[T, M[S]]