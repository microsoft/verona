# LLVM Lowering

After the last MLIR pass, the IR will only have standard dialects operations
and Verona dialect operations that cannot be lowered to other dialects. The
final full conversion to the LLVM dialect will therefore have a number of
special lowering from specific Verona operations.

## Assumptions

The main assumption at this point is that all control flow operations will
have already be converted to standard dialects, including dynamic calls (and
their respective lookups for the address), even if some of the intermediate
steps are still in Verona dialect.

This also means all basic blocks and branch instructions will already have the
necessary arguments, the control flow graph will be correct and representing
user code and potential optimisations accurately.

All classes will be concrete (including `when` blocks, lambdas, constants) and
represented as MLIR `module`s for lexical context, their methods will be inside
the `module`s at the right place and their fields will be declared in the type.

All vtables and runtime data structures will have been lowered already as struct
types, with the type/field offsets baked in the MLIR lowering stage as plain IR.

Finally, the interaction with the Verona runtime will have already been defined
and encoded as calls to library functions (scheduling, memory allocation, etc).

In summary, the full LLVM dialect conversion should be as trivial as possible.

# Full LLVM dialect conversion

## Classes

LLVM has native struct types. Literal types are uniqued structurally (identical
layout references point to the same unnamed type). Named types don't merge and
can be used to represent different classes even if they're structurally the same.

However, struct fields can only be valid LLVM types, so we need to convert all
Verona types into LLVM types and appropriate functions to handle the semantics.

For example:
 * Class types are just LLVM struct types, recursively. To avoid declaration
   order issues, a type can be forward-declared as `opaque`, and then fully
   defined later.
 * Pointer/reference types need to be pointer types in LLVM, to some other
   struct types. There may be some extra handling of the pointers (for instance
   increasing the reference count, cleanup after last use), but those are already
   part of the ABI and will have been lowered earlier in the IR.
 * "Native" types (integer, float) are still refering to a class (ex. U32) and
   should be converted to appropriate LLVM types (ex. i32 with unsigned
   semantics), and the subsequent operations will be converted to native LLVM
   operations (ex. add).

All other types should have been converted into a sequence of the types above and
appropriate operations on them, including calling the runtime library.

MLIR `module`s could have a class type (`!verona.class<...>`) which describes the
fields it has, that will be lowered as LLVM struct types, for storage context.

## Types

Throughout the passes, pointer capabilities (such as `iso`, `mut` and `imm`)
would have already been converted into compile-time checks (and failed if wrong),
runtime checks (with calls to the runtime) or special lowering (sequence of
instructions) on the code itself.

This means by now we don't need those annotations and can drop all but the
concrete types. Those would be concrete class or pointer/reference types to
concrete classes, which would be indexed by name and we can just reuse the
already declared class types via their LLVM struct types of the same name.

We will probably need some mangling to get unique names across all classes and
modules.

### Regions / Cowns

By now, all region information has already been converted to optimisations on
the existing code, alias annotations on the remaining operations and calls to
the runtime for scheduling, cleaning up, etc.

There is nothing left to need special convertions.

## Functions

All functions, including class static and dynamic methods and compile-generated
helpers, will be in the form of an MLIR function, with concrete typed arguments
and full standard dialect control flow.

Those can be directly lowered into the LLVM dialect on their own (via the
dialects' own converters). Additional Verona dialect operations that exist will
have their own implementation and be trivially converted.

## Statements

### new_region / new_object

Those can be directly lowered to calls to the runtime to manipulate memory and
control access and shouldn't have any special representation left in the
Verona dialect.

### call / static

By now, all static calls have already been converted to a standard `call` to
existing (mangled) functions and all dynamic calls have already been converted
to `call_indirect` with the address calculated via vtable lookup.

These lower to LLVM dialect directly and need no special convertion.

### when

`when` statements have already been converted to a sequence of creating the
cown and calling the schedule runtime, which later on (see above) will be
converted to standard calls, so again, nothing special here.

### field_read / field_write

All types at this stage are complete and concrete, which means reading/writing
from/to fields in LLVM are just direct access to struct fields (ie. via GEP).

### numeric literals

All numeric literals, including compiler generated ones, can be represented
directly in LLVM. However, in MLIR, they have been lowered as an `imm` anonymous
class, wich access control (ex. reference count) and thus calls to the runtime
library or lowered directly as code.

If we want to optimise some literals over others, we may need to do so before
the lowering to LLVM, or it would create the arduous work of cleaning up
previously lowered code.

Their values, however, would have a constant initialisation that will be lowered
as a global value of the same type, with the name mangled via the anonymous
class' name.

### string literals

Similar to numeric literals, string literals will be represented as anonymous
classes and will have the initialisers as global array variables.

The type of the array may depend on the actual encoding (i8, i16, i32) and will
be decided at the LLVM conversion time, mangled by the anonymous type and the
number of strings in the same type (for interpolation).

### match

All compile-time type matches will have already been lowered by now, and either
elided or specialised, so any remaining `match` is a runtime operation to check
the type of the variable.

LLVM does not support multiple return values, so we may need to call the runtime
with an extra parameter (the value pointer) to be filled in case the match is
successful.

### end-scope

These would have already be converted to calls to `release` and therefore don't
need any additional logic.

## Builtins

Depending on what we do with the arithmetic passes, we may endup with calls to
LLVM builtins (as "native MLIR opaque calls"), and therefore can be directly
lowered to calls to named symbols which LLVM can resolve directly.

Example:
```
  // MLIR:
  %0 = "llvm.some_special_add"(%arg0, %arg1) : !verona.U32
  // LLVM dialect:
  %0 = llvm.call "llvm.some_special_add"(%arg0, %arg1) : i32
  // LLVM IR:
  %0 = call i32 @llvm.some_special_add(i32 %arg0, i32 %arg1)
```

It's up to LLVM to decide what to do with the builtin, either call a runtime
library function or try to lower to some sequence of instructions that are valid
on the target architecture.
