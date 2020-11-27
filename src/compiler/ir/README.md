# General structure

The IR has a fairly standard structure, made up of basic blocks. Each basic
block has a list of statement and a terminator. While the implementation
occasionally refers to "phi nodes", the IR uses basic block arguments rather
than phi nodes.

The IR only represents the body of methods. The "shape" of the program, eg. its
classes, fields, method names and signatures, ... are all stored in the AST
instead. For each `Method` node in the AST, a corresponding `FnAnalysis` is
created, which, among others, contains the `MethodIR`.

A `MethodIR` is composed of one or more `FunctionIR`. This distinction is used
to encode the bodies of `when` clauses that appear within the method. The first
`FunctionIR` corresponds to the method body itself, whereas subsequent blocks
correspond to when block bodies.

Once created by the `ir/builder` pass, the IR is immutable. All new information
is represented in external side structures, as part of the `FnAnalysis`
structure.

# Type information

The IR itself contains almost no type information. Only types which appear in
the method body at the source level is included in the IR. At the moment, this
is just types used in a match block.

The inferred type of IR values is stored in maps in `FnAnalysis`. The type of
`v` in basic block `bb` can be found in `analysis->typecheck->types[bb][v]`.

Similarily, at construction, every method call site is assigned a
`TypeArgumentsId`, used to identify the call's type arguments. Given an `id`,
the inferred type arguments can be found in
`analysis->typecheck->type_arguments[id]`.

In both cases however, these are the polymorphic types. If a method or class is
generic, these types may mention type variables. The non-generic types will
only be known after reachabilitiy and reification.

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
as this is the instiation for the class that has been inferred by type inference, and 
```
   Foo[double]
```
as this is the inferred return type.  For the method, we will look into there bodies to see what types they use,
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

# Region Tracking

We need the provenance of each mutable reference to an object to its 
region to be represented in the code.  
This is only required for variable that can contain a mutable reference into region tree.
We need this because we support certain operations on a region that can be done by having
a reference to any particular object in that region.

An `iso` reference is both a handle on an object, and can cheaply access its region
information.

A `mut` reference does not cheaply know its owning region, 
so we need the compiler to have a second
reference associate with each `mut` that refers to its region.

To maintain this data, for an aliasing field read
```
y = x.f
``` 
we need translate this to
```
y = // Read field described below.
if (y->rt::Object::get_class() == RegionMD::ISO)
  y_region = y
else 
  y_region = x_region
```

And for extracting field read:
```
y = (x.f = z)
``` 
similarly has
```
if (y->rt::Object::get_class() == RegionMD::ISO && y != x_region)
  // Extracted iso, so the region pointer is not required.
  //  By using null, we enable pattern matching on extracted isos.
  y_region = null
else
  y_region = x_region
```
This differs slightly as it also keeps enough information to determine
if a reference is an `iso` capability.  The `y != x_region` check is
required because `mut` can reference the entry point of its current
region, and that should not be treated like an `iso`.

The introduced sequences are good candidates for MLIR optimisations as type information
can be used to simplify them.

This can be used in object construction as follows
```
  x = new C in y
```
should use `y_region` to call the runtime for the allocation.

Also, this can be used to pattern match to resolve `iso | mut` capabilities:
```
y = (x.f = z)
match y with
  iso => C1
  mut => C2
```
Would become
```
y = x->f;
x->f = z;
if (y->rt::Object::get_class() == RegionMD::ISO && y != x_region)
  // Extracted iso, so the region pointer is not required.
  y_region = nullptr
else
  y_region = x_region
if (y_region == nullptr)
{
  [[C1]]  // Lowering of C1
}
else
{
  [[C2]]  // Lowering of C2
}
```

This is particularly important for commands like `drop` that deallocate a region, where the parameter
may or may not be a region:
```
// x : (mut|iso) & ...
drop(x)
```
would be translated to
```
if (x_region == nullptr)
  // call into runtime to deallocate `x`
```

The region assoicated with each `mut` also affects the calling convention.
Each argument that can contain a reference of type `mut`
needs to be made into two arguments, the region entry point and the object itself.  Similarly,
return types that can contain `mut` must be double width to encode the region entry point
of the returned values.

Many places the region associated with a variable will not be required, and dead code elimination 
can be used.
A global analysis could also be implemented to remove unneeded region parameter and returns for functions/methods.


Aside:  We do not have to worry about allocating on a null region
variable. The front-end will perform sufficient work such that
```
  x = new C in y
```
will guarantee that `y` is not an `iso` by adding coercions to
`mut` such that all accesses in a function will determine they 
are part of the same region
```
  let x1 = new C in y
  let x2 = new C in y
```
The type system will ensure that `x1` and `x2` have the same
lifetime type (assuming we go that way).

# Statements
## `NewStmt`

Allocates a new object of a given class. It may optionally be given a `parent`
which designates the region in which the object should be allocated. If no
parent is specified, the object is allocated in a new region.

This will correspond to either a `rt::RegionTrace::create` or
`rt::RegionTrace::alloc` call, dependending on whether a new region needs to be
allocated.

The `parent` specified in the statement could be any object in the target
region; it may not be a reference to the region's entrypoint. On the other
hand, the runtime's `RegionTrace::alloc` method expects to be given the
region's entrypoint. The compiler therefore needs to do some work to find the
entrypoint given the specified parent.

## `CallStmt`

Call a method, on the specified receiver. The method is identified by its name.
At compile-time, the method's selector should be obtained from the given name.

In order to dispatch the call, in the general case, the selector should be used
to index into the object's vtable, to retrieve the address of the target.
Optimizations may enable us to dispatch the call statically, using for example,
the type of the receiver.

The method being called could be static or not (ie. does it accept a `self`
parameter, not to be confused with the dispatch strategy). In either case, the
receiver of the call is always passed as the first argument to the method.  In
the case of a static method call, the receiver may not be an actual object, but
only a pointer to a descriptor, as created by a `StaticTypeStmt`.

## `WhenStmt`

Schedule a behaviour on a set of cown. The `closure_index` corresponds to the
index of the body within the `MethodIR`.

This corresponds to a `rt::Cown::schedule` call. The message sent to the cowns
should provide storage for a pointer to the closure's body, the list of cowns
and the captures.

The closure's body will accept as many parameters as there are cowns and
captures specified. When invoking the body, each cown argument should be
transformed into a reference to its contents, before being passed as an
argument. The captures should be passed as is.

## `ReadFieldStmt`

Read a field from an object. The field is identified by its name. The general
case is similar to method calls: the field's selector is used to look up the
field's offset in the object's vtable.

## `WriteFieldStmt`

Write a field into an object. The procedure for locating the offset of the
field is identical to a field read.

When writing a reference counted value to an object in a region that is
equipped with a RememberedSet, the value should be inserted into the RS.
Otherwise, the reference count should be incremented.

## `IntegerLiteralStmt`, `StringLiteralStmt`, `UnitStmt`

These create values from a literal. The support for strings is very special
cased for demonstration purposes.

## `MatchBindStmt`

This is an unchecked cast, used at the top of the target of a match statement.

## `ViewStmt`

If the argument to the statement is an owned value, the statement produces an
unowned reference to the same object. Otherwise this is equivalent to a copy.

## `EndScopeStmt`

This statement is inserted whenever variables go out of scope. This may not be
the end of a BasicBlock, if a block expression is used for example.

This is where a value's "release" code should be executed. This could be
`rt::Region::release`, `rt::Cown::release` or `rt::Immutable::release`, for
owned, cown and immutable values, respectively. Unowned values do not need to
be released.

## `OverwriteStmt`

This is used after a local variable assignement, on the old SSA value bound to
this statement. The old value should be released, as in an `EndScopeStmt`. For
example, the code `var x = new A; x = new B; ...`, would be lowered to the
following IR:

```
0 <- new A
1 <- new B
overwrite(0)
// ...
end-scope(1)
```

The local `x` is originally assigned the SSA variable `0`, and later `1`. After
the assignement, the `overwrite(0)` statement encodes the fact that the old
value needs to be released.

Note that an SSA value that has been overwritten is never passed to an
"end-scope" statement. Only `x`'s new value, `1`, goes out of scope.

# Terminators

## Branch Terminator
## If Terminator
## Match Terminator

# Builtins

A number of language features don't need special syntax, but are instead
exposed as builtin methods, defined in `stdlib/builtin.verona`. This includes
arithmetic operations and some region operations such as freeze and trace.

The language features they represent isn't even expressible in the IR. Instead,
`codegen/builtins.cc` provides hardcoded lowerings into bytecode.

