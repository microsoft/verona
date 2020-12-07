# MLIR lowering

The output of the front-end will be lowered into MLIR as a mix of dialects,
including a Verona specific one. How much of each will depend on the support
of the existing dialects to Verona's own concepts.

The long term plan is to have a single IR (we call 1IR), based on MLIR, with
a distinct Verona dialect for type inference, free variable and liveness
analysis, reification and ellision based on type information, region based
alias analysis and optimisation, dynamic calls optimisation (descriptor
colouring), etc.

However, the complexity of each pass and the distributed development nature have
risen design issues that are hard to solve concurrently on the same dialect. So
we are delaying the construction of a complete dialect in favour of having two
IRs (high-level and low-level) for the time being.

The high-level IR is described here: src/compiler/ir/README.md

This document is about the low-level IR, which will be converted from that
high-level IR for now. In time, that IR will be converted to operations in the
Verona dialect that will be partially lowered to the onces defined here in order
to be further lowered into LLVM IR.

In this document, we'll call the high-level IR "1IR" and the low-level IR "2IR".

## Assumptions

The main assumption is where the cut in passes will be done between 1IR and 2IR.

The main driver for the 1IR is to do type analysis, which mainly means
type inference and reification, which need free variable and liveness analysis.

So the passes that will be done in MLIR for now are region based alias analysis
(which needs concrete types and region information) and descriptor colouring,
which needs all classes and their methods to be known, as well as all dynamic
calls to be explicit.

Literals should also already be lowered to their corresponding anonymous structs
including string representations (see #295), as they may have to change
depending on type analysis of objects interacting with them.

Lambdas and functional objects should also have been lowered to their anonymous
structs with the correct captures and concrete types.

## Requirements

### No information should be left in the AST

Long term, we should convert IR to IR only. This will make it much easier to
later convert the 1IR into 2IR dialect operations, allowing for partial
conversions.

However, as a first approximation, it should be fine to have to walk the AST
to infer class / module structure. The current IR cannot represent those concepts
and the code structure remains in the AST, with pointers to IR block sequences
for function body, lambdas, `when` blocks, etc.

### Make minimal use of foreign support structures

While both IRs will use maps and trees to keep information about the code being
transformed, we should try to isolate the result of analysis that can be
invalidated by following transformations.

Ultimately, the passes being done in the 1IR will have to be migrated
to MLIR passes and will be implemented as analysis or transformation passes with
the exiting structure.

We should keep in the IR as much as we can in the form of annotation, traits and
types, so that any generic pass can transform it without having to resort to ad
hoc compiler data structures.

This may not be a requirement of the 1IR now, but as we move the
boundary up, implementing each last pass of the 1IR in MLIR and
raising the level of the 2IR Verona dialect opreations and types, we'll need to
make that distinction. Therefore, it will be easier if we implement at least the
interface between them as such now.

### Simple control flow

To make full use of the existing dialects, we need to have concepts that can be
lowered to them in simple ways.

Calling objects should be only:
 * Static calls should really be a `std.call` to existing functions, with
   mangled names and inside visible (MLIR) context.
 * Dynamic calls should be a pair of `get the descriptor` and `call the address`,
   neither of which are control flow at the MLIR level and will probably be
   dialect ops with direct lower to LLVM.
 * Lambdas and functional objects should have a struct with captures and a
   static function to call.

Conditionals and loops should be only:
 * if/else as a `std.cond_br` into one or two blocks with an exit block.
 * Loops already converted to `while` loops in the form: prep/cond/body/tail.

`when` blocks should have already be converted to lambdas with `cown`s as captures.
They may also need some preparation for transforming it into a message, for
example, `create message(lambda); dispatch(msg)` as semantic nodes. We can
convert those nodes to library calls in MLIR directly.

The message handling could be done at the MLIR level, but we still need a node
to make the distinction between a user lambda and a `when` lambda. This could be
a simple `when` node pointing to a lambda, but we need the lambda to have already
been lowered, because all others are and we don't want to duplicate that code.

`match` blocks that can be known at compile time will have already been simplified
in each version of template (partial?) implementations as well as restrictions
from type inference.

All the remaining matches are dynamic and may need a dialec operation until it
can be lowered to the appropriate reflection calls to check the type, when it
will turn into a simple if/else block.

### Excplicit runtime library entrypoints

Part of the language semantics will be implemented by the runtime library. Some
of it will be known at the high-level scope (AST, 1IR) while some of it only
after enough passes have gone, or by the time we're lowering to LLVM IR.

The knowlege of *how* to lower is up to each level, but the knowledge of *where*
can be decided by much higher levels, and leave hints to the lower levels in the
form of AST nodes, IR or dialect operations.

If the calls depend on the type, for example which type a literal string will be
stored as (ascii, UTF-8, etc) will change which functionality to call from the
libraries and even the method names, which may not be known at a lower level.

A clean implementation, and in the spirit of MLIR, is to have implementation
details as dialect constructs (ops, types, attributes) that can later be easily
lowered to a call to a library function using the available information.

If the code is elided, no information is kept, no harm done. If the types change,
the information is carried with the type, and the lowering is still trivial.

# Structure

## Overall structure

The current 1IR design has the AST as the main structure, with the basic blocks
in IR form. This should change to a pure IR representation in time, but even
before that, we can use the AST class structure to lower structs for fields and
mangle names for functions only, with the rest lowered directly from 1IR as the
body of methods.

Class and module lexical context will be controlled with MLIR's `module`
construct, with their regions containing a list of methods that belong to that
class/module.

Functions will be lowered as MLIR functions, called with `std.call` for static
functions that are resolvable via MLIR symbol table and `verona.call` for the
dynamic calls that will later be converted into vtable lookups (made concrete
only after colouring).

Lambdas and `when` blocks will have already been converted into an anonymous
class with captures as fields and the `apply` method as the body. Constant
literals will also be lowered in that way, with the initialisation as the
`construct` method and the value as an immutable field.

Due to reachability analysis being done in the 1IR, by the time we get it to
lower to 2IR we won't have any generics. All methods and fields will be
guaranteed to be used/called in the program from the entry point and all the
types will be concrete and correct.

## Function Bodies

Function bodies have multiple `BodyIR` regions, one for the main body and one
for each `when` clauses. The main body will be lowered in the context of the
encompassing class (struct field access, mangling) while the `when` bodies will
by lowered on their new anonymous classes and captures, still within the same
context, so to guarantee no external user can access it by mistake.

## Basic Blocks

So far, there is only two constructs that create new basic blocks: conditionals
and loops. Both will end with a tail block that will be the main block of the
subsequent operations.

The main issue when lowering basic blocks is figuring out their arguments. For
this we need free variable and liveness analysis, which will have already been
done by the 1IR passes. That information is in the `FnAnalysis` structure for
now and it should be trivial to use it for arguments without having to backtrack.

Once the 1IR becomes more complete (with less side structures), we'll need some
information in the IR itself (condition/loop nodes).

## Types

All types in 1IR should already be concrete at the time of conversion, so all
Verona types in the 2IR must validate using standard MLIR checks, which only
check for equality, not similarity or sub-types.

If there are any cases where this is not true (in != out), we can add casts in
expectation that the type check in 1IR was successful and therefore the cast is
valid.

Class types should be lowered to a `!verona.class<fields:types>`. We don't need
a similar struct representation, since every field read/write should have the
class type and the field name, with the concrete field type as a result.

## Regions

Every object is allocated into a region. Some objects share regions, others
live in immutable memory. Whether we'll allow users to annotate objects with
region information (ex. `new Foo in Bar`) or not is up to debate. Regardless,
we'll need that kind of information in the IR to add calls to the right runtime
library functions that control memory.

We want a pass that bundles objects into regions based on their access patterns
taking user annotations as either hints or requirements. Such pass could be done
in 1IR or 2IR, but it will depending where it runs and whether we allow user
annotations will dictate if the 1IR will have region information, but the 2IR
will definitely need it.

Region informaiton can be represented as references to the owning `iso` (see
below) on the types on every instance of, to avoid repeated and potentially
complex search patterns.

### ISO

Iso references are region owning pointers that can freely access the whole region
and sub-regions (if not given to another task). Any read-write access through
those references is free.

### MUT

Mutable references must belong to a region and must have access to that region,
ie. they need to belong to the owning `iso` of the region. For this, mutable
references must point to the `iso` references. This can be represented as
MLIR annotation, referring the SSA value (ex. `#region<%22>` where `%22` must be
an `iso` reference of the same region).

### drop

Dropping (deallocating) regions need to know that there are no more active users
of that region (ex. reference counting) and also need to find which region it is.
If we're dropping an `iso`, it's straight forward, if it's a `mut`, we simply
follow the annotation to find the `iso`.

The actual deallocation will be a call to the runtime library if the region is
gone (ie. equals `nullptr`). This is a simple check that can be lowered by the
translation directly.

## Statements

### NewStmt

Allocates a new region or more memory in an existing region. We should keep
this as a dialect operation until we do region optimisation. This is not the
same as region tracking in 1IR, which will validate regions and insert `merge`
nodes where necessary.

The `NewStmt` will be lowered as `verona.new` and merge nodes as `verona.merge`
and both will allow us to optimise region merges using flow analysis (and other
tricks) to merge regions earlier on (replacing `merge(a, b)` with `new b in a`,
for example).

The operation has two semantics plus the merge:
```
  // new object in a new region
  %0 = verona.new_region @Class [initialiser]

  // new object in an existing region
  %1 = verona.new_object @Class [initialiser] in @Other

  // Merge two regions
  verona.merge %0, %1
```

### CallStmt / StaticTypeStmt

Calls a method via a descriptor. The method is passed by name and the descriptor
can be either a static type reference (for static methods) or an object of some
type (for dynamic dispatch via vtable).

To get a descriptor, we either take the SSA value from the variable reference or
we use a dialect operator to get a static descriptor from a type. Then the call
is just an operation on that value and the method name, with arguments.

The static descriptor is created via `verona.static` and the call is done via
`verona.call`, with arguments being the descriptor and all operands. The calls
are expected to have been converted already from `infix` to `prefix` (ie. `a+b`
to `a.+(b)`).

The syntax is roughly:
```
  %0 = verona.static !verona.class<Foo>
  %res = verona.call "method" [%0] (%arg0, %arg1...)
```

Static calls can be further lowered to an `std.call` to a mangled function name
(example: `a:U32 + b:U32` -> `a.+(b)` -> `U32.+(a, b)` -> `__U32_add(a, b)`).

Dynamic calls will need to find the offset of the method's address in the vtable
and call that address, which is harder to optimise as an `std.call`. We may
still be able to inline the call if we can find the method and if the
arguments and types are all known and monomorphic, which will probably need to
be done before lowering to LLVM, as we have type and region context at the MLIR
level.

The syntax is similar to the static call:
```
  %0 = something that gives an runtime object of class Foo
  %res = verona.call "method" [%0] (%arg0, %arg1...)
```

Fields and variables that can be called should point to the correct method's
address so no vtable lookups would be necessary on call, as they've already
been done on assignment. However, it may be harder to know to which method they
refer to, so harder to inline, even with inter-procedural analysis.

The syntax is similar to the static call:
```
  %0 = something that gives an runtime object of class Foo
  %res = verona.call "method" [%0] (%arg0, %arg1...)
```

With `method` being a field name, not an actual function/method. Depending on
how hard it is to discern this at compile time, we may want to change the syntax
of the operation (or create a new one) to make it clear it's a field, not a
method.

Lambdas are anonymous structures already initialised by the declaration, with
an `apply` method, so the call mechanism is exactly the same.

### WhenStmt

Scheduled behaviour as a lambda initialised with the captures as usual. The
`apply` arguments are the `cown`s in the `when` clause. In additional to the
lambda, a `when` statement also calls the runtime library to schedule the
behaviour (push it to the work queue).

So, a `WhenStmt` is roughly lowered to the sequence:
```
  std.call Anon$123.create(capture1, capture2,...);
  %descriptor = verona.static(Anon$123);
  std.call rt.Cown.schedule(%descriptor, %cown1, %cown2, ...);
```

### ReadFieldStmt / WriteFieldStmt

Access class fields for read/write. The location of the field and the operations
to extract them are defined in the ABI, which hasn't been calculated yet, so
we need dialect operations to keep the information on the field name until the
ABI pass converts field access into lowering patterns.

The operation is similar to method calls:
```
  %0 = something that gives an runtime object of class Foo

  // Read returns the field value
  %res = verona.field_read "field" [%0]

  // Write returns the previous field value
  %new_val = constant 10
  %old_val = verona.field_read "field" [%new_val]
```

There are different rules for accessing single typed fields and union types. The
current ABI proposal lays out in order of size, from largest to smallest, and
therefore the offset from the object's address is not the same as the sum of all
previously declared (in program order).

Furthermore, union types may have compacting optimisations and discriminators to
match which value is stored, so will need non-trivial extraction lowering (mask
and shift on the discriminator, then mask, shift and possibly extend on the
payload). Writes will have to do the reverse process to save values in the ABI
representation.

### <Integer|Float>LiteralStmt

Constant literals with generic type (int, float). The type we assign to them
will depend on how they're used and therefore will depend on type inference and
reachability analysis, which should have been completed by the time the 1IR is
ready to be converted in the 2IR.

Literals are lowered as immutable objects of the specified concrete type,
initialised with the literal value.

Example:
```
  // 10 + a : U32
  %const = verona.constant "10" : U32
  %0 = verona.new_object @U32 [ %const ]
  %1 = verona.call "+" [ %0 ] ( %a )
```

### StringLiteralStmt

Strings are complicated on both Verona and MLIR.

Verona's model is converging to one where strings can have different
representations but a unifying interface, so that concrete string types can
coexist and be converted to one another. This adds lowering as well as
representation complexities.

Furthermore, strings can be interpolated with values. If the values are
compile-time constants, then their string representation can be expanded at
compile-time. If not, we need to splice the string into a list of literal +
value + literal, etc. We may need to do that _before_ the concrete representation
is chosen.

MLIR doesn't support strings at all. We'll have to keep their values in some meta
storage until we can lower them into LLVM. LLVM IR supports string literals and
we can call string maniupulation library functions to operate on them. The MLIR
Toy example has a mock implementation of that process.

Example:
```
  // Simple string
  %simple_string = verona.constant "This is a string" : !verona.string

  // Interpolated string ("My name is $1 and I'm $2 years old", name, age)
  %0 = verona.constant "My name is " : !verona.string
  %1 = verona.constant " and I'm " : !verona.string
  %2 = verona.constant " years old" : !verona.string

  // This is still not decided, but just for illustration purposes
  // We could also pass this to the representation, so it can do its own thing
  %name_str = verona.call "toString" [ %name ] : !verona.string
  %age_str = verona.call "toString" [ %age ] : !verona.string

  // This is whatever runtime string representation we choose
  // assuming we have a constructor with a list of literal strings
  %utf8 = verona.new_object @UTF8String [%0, %name_str, %1, %age_str, %2]
```

In the case above, if all variables are already compile-time constant strings
at the time of lowering, the compiler has the ability to fuse the strings,
creating a simple string (or at least less splits).

### UnitStmt

No idea.

### MatchBindStmt

An unchecked cast for runtime `match` statements. If the type of the runtime
object isn't compatible with the cast, an invalid object returns?

A simple way to lower `match` statements is to chain basic blocks with
conditional branches.

```
  %val = the runtime value passed to the `match` statement
  %0, %val0 = verona.match @Type1 [ %val ]
  cond_br %0, ^bb1(%val0), ^bb2
^bb1(%local_val : Type1):
  // Implement `case Type1`
  br ^lastbb

^bb2:
  %1, %val1 = verona.match @Type2 [ %val ]
  cond_br %1, ^bb3(%val1), ^bb4
^bb3(%local_val : Type2):
  // Implement `case Type2`
  br ^lastbb

  ...

^bbX:
  %y, %valy = verona.match @TypeY [ %val ]
  cond_br %y, ^bbY(%valy), ^lastbb
^bbY(%local_val : TypeY):
  // Implement `case TypeY`
  br ^lastbb

^lastbb: // no args

```

LLVM IR has support for `switch` statements, but they require the
type of the object to be an integer. We could create a dispatch table and use
offset in that table as the switch's value. But this would be an optimisation
that has no clear benefit for now.

### ViewStmt

Created a mutable reference to an object. The validity of the type system has
already been checked by the compiler at this point, so all reads and writes to
those values are valid.

At the 2IR level, there is no need to represent this statement as an operation
and the only implementation detail is to add the previous value to the new
symbol.

### EndScopeStmt / OverwriteStmt

When variables go out of scope, either lexical or type related (ex. move
semantics). We need to call `release` on owned region types (region, iso, imm)
but not on unowned types (mut).

Example:
```
  %0 = some region

  ... use region

  // Here we get EndScope %0
  %rt = verona.static !verona.class<rt::Cown>
  verona.call "release" [ %rt ] ( %0 )
```

This will do what's necessary depending on the type of region, for example,
decrease the counter or deallocate the object altogether.

`OverwriteStmt` has the exact same semantics but identifies an object which
ownership has been moved, not killed. This distinction is irrelevant to 2IR.

## Builtins

Calls to the runtime library that have already be lowered byt the 1IR, so
no extra work is generally needed.

Some of those builtins are usually one-liners and will be inlined by further
passes in the compiler (likely LLVM). Other calls (maybe arithmetic?) could
be lowered specially to direct LLVM IR constructs at the final conversion step.
