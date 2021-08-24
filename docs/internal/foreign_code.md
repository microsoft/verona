Verona Foreign Code Interop Design
==================================

The Verona language does not exist in isolation.
Rewriting all of the useful C/C++ code that could be written in Verona would cost trillions of dollars and so is not a feasible option.
Languages such as TypeScript and C# provide mature application development ecosystems that we wish to enable on top of Verona services.
Verona will have to exist in a world with other languages but that should not compromise the security guarantees that we aim to provide.

Core principles of interoperability
-----------------------------------

The Verona interoperability model will be built on the following principles:

 - Using foreign code should not require giving up on the safety guarantees of Verona.
 - Interoperability should not compromise any security guarantees of the non-Verona language or library API.
 - Most libraries are stateful, the correct granularity for interoperability is a library, not a function.
 - Users should not have to write any boilerplate to use a foreign API.

The first of these relates to one of the core principles of Verona: there is no `unsafe` keyword.
Verona targets security-critical workloads and it's very hard to make strong security claims about a language if any call to another language is able to compromise any and all of our safety guarantees.
The Java JNI specification, for example, contains this disclaimer in the Design Overview chapter:

> The programmer must not pass illegal pointers or arguments of the wrong type to JNI functions. Doing so could result in arbitrary consequences, including a corrupted system state or VM crash.

The language makes an explicit declaration that any call to native code can undermine any of the invariants that the Java security model depends on. 
This is a pragmatic decision to allow quick adoption but it means that no Java program can be considered safe until all native components have been removed.

Modern C++ codebases work hard to provide memory safety guarantees at the API level.
These guarantees are often lost in interop layers.
For example, LLVM makes extensive use of typed smart pointers in the C++ APIs, but the Python bindings are written on top of the C API, which make type safety the caller's responsibility.
Even if we can sandbox the resulting code, using a C++ library from Verona should not be more likely to introduce security vulnerabilities than using the same library from C++.
The 2017 paper [*Finding and Preventing Bugs in JavaScript Bindings*](https://ieeexplore.ieee.org/document/7958598) showed that memory safety bugs in the C++ / JavaScript interop layer in Chrome are surprisingly common.

Any time that a user has to hand-write code at a bindings layer to get memory safety right, we can assume that they will get it wrong and so we aim to automate as much as possible.
Users will wish to write wrappers that expose Verona-friendly idioms for foreign libraries but they should never have to duplicate any information that exists in a C/C++ header file.

The desire for sandboxing also leads to the conclusion that a 'foreign function interface' is exposing the wrong level of abstraction.
What sandbox should a function run in?
Our prior [CHERI-JNI](https://www.cl.cam.ac.uk/research/security/ctsrd/pdfs/201704-asplos-cherijni.pdf) work showed that treating libraries (in the loose sense, including any dependent libraries) as the unit of foreign code provided an abstraction that fitted well with sandboxing and allowed the user to determine the degree of state sharing between foreign code invocations.
This abstraction also has software engineering benefits.
LLVM, for example, does not have stable ABI (or API) guarantees across programs and so can cause linking errors if, for example, an application depends on one library that uses libclang from LLVM 10 to parse syntax highlighting and another that uses the JIT from LLVM 11 for GPU shaders.
In a Verona world, these would be in different sandboxes with different 'global' namespaces and so would not conflict.

### Sandboxes are surfaced as regions

Sandboxing is a security *policy* that can be implemented with a variety of *mechanisms*.
Sandboxing runs a component with a reduced set of privileges and limits its interactions with the outside world.
It can be enforced with a separate process that runs with reduced OS privilege (e.g. the OpenSSH privesp model or Chromium's renderer sandboxes), with SFI techniques such as those used in WebAssembly compilers or NaCl (potentially with hardware acceleration from features such as Intel MPK), or with in-address-space hardware isolation such as CHERI.

Sandboxes are difficult to program with in C-like languages because there is no useful construct in the abstract machine that can represent them.
Web browsers are the most aggressively compartmentalised applications and yet run with a handful of different sandbox types.
Because there is no abstraction in C for distinct sets of objects with mediated entry points, sandboxing code written in C leaks mechanism into the security code (for example, the idea that sandboxes are process).
Any call between sandboxes must go via some RPC layer.
Allocating memory inside a sandbox from outside and directly manipulating data structures is difficult to get right because the language doesn't impose any restrictions on where any given pointer can be stored.

A Verona region is a set of related objects, with arbitrary topology, that have a single entry point.
Sandboxes map cleanly to this abstraction.
A sandbox's heap can contain pointers to any other data structures owned by the sandbox.
It cannot contain pointers outside of the sandbox because following them would violate the sandbox policy.
The Verona type system enforces this automatically by disallowing cross-region pointers.

We hope that the region abstraction will make programming with sandboxes natural within the language.

### Foreign libraries are classes

If the unit of interoperability is a library, how do we surface that within Verona?
We aim to treat foreign libraries just like any other module: as classes that can contain nested types.
A package is referred to in Verona by a path.
Normally this refers to a directory that contains Verona source code.
For foreign modules, this provides a configuration file that contains (for C/C++):

 - The header file describing the interface.
 - The compiler flags required to parse the header file (for example, predefined macros, language dialect).
 - The library file containing the implementation (initially this will be a .so / .a file, it may be a .wasm file for WebAssembly sandboxing once that is implemented).
 - Any other metadata that we require (TBD).

This file should be trivial to generate from existing build systems.
Once C++ module support is more widespread, module imports may be able to take the place of header inclusions.
The Verona compiler is responsible for parsing the interface description and mapping it into Verona constructs.

For example, a Verona module using a C++ library that exposes a CMake target (for example, one installable via vcpkg) might include an interface description `verona.foreignpackage.in` something like this:

```json
{
  "name" : "libfoo",
  "link_flags" : "@LIBRARY@",
  "include_file" : "libfoo.h",
  "language" : "c++",
  "dialect" : "c++20",
  "compiler_flags" : "@CMAKE_CXX_FLAGS@"
}
```

The CMake build system would include a line something like this:

```cmake
find_package(libfoo CONFIG REQUIRED)
get_target_property(LIBRARY_DEPS libfoo::libfoo IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE)
get_target_property(LIB_FILE libfoo::libfoo IMPORTED_LOCATION_RELEASE)
set(LIBRARY "${LIBRARY_DEPS} ${LIB_FILE}")
configure_file(verona.foreignpackage.in verona.foreignpackage)
```

This is an incomplete example.
A more complete example collects all of the compiler and linker flags for the imported target and propagates them to the JSON file.
It should be possible to provide a generic CMake function that extracts everything required for an imported library, or integrate this directly into vcpkg so that a Verona module just needs to specify the vcpkg ports that it wishes to use and provide a header file that includes the headers that should be exposed. 

Mapping from C constructs to Verona
-----------------------------------

Most foreign code layers work well with C code and, although our goal is rich C++ interoperability, since C provides a subset of the constructs in C++ it's a good idea to start here and look at some things in the common subset separately.
A C library exposes an interface containing structure and type definitions, global variables, macros, and functions.
Each of these provides some difficulty for interoperating languages.
For example:

 - Structures may have attributes that define packing or padding properties or alignment requirements.
 - Structures may contain bitfields, which have platform-specific lowering requirements.
 - Functions may specify different calling conventions and even with the common ones the lowering from C to LLVM IR is non-trivial and platform- and architecture-dependent for any types that are not integers or pointers.
 - Static inline functions in headers don't exist as callable symbols in the generated library and so can't be called unless the foreign-code layer can compile arbitrary C code.

We consider macros out of scope for MVP, though 1.0 should support macros that trivially evaluate to constant expressions.
Programmers can work around this in the interim by defining static globals that evaluate to macros in the header that defines the library interface.
For example:

```c
#include <fcntl.h>
static const int o_rdonly = O_RDONLY;
```

This header is still portable (it will pick up the platform's definition of `O_RDONLY`) but it is a small amount of extra boilerplate.
This is more annoying where macros correspond to identifiers, for example in common implementations of `siginfo_t`.
This limitation can be worked around by providing accessor functions in the interface header.


### C types

C structures and unions are surfaced as nested classes within the class that is exposed for the library.
Each field may be accessed via the selector with the correct name.
All fields are treated as `mut`, even if they are declared as `const` in the C header (all C/C++ type annotations are treated as advisory, since C/C++ are not type-safe languages).

Classes implemented as C structures do not correspond to any interface.
**Note** It may be possible to relax this at some point, though it's not clear that this is desirable.
C structures do not contain a type descriptor and, even if they did, it cannot be trusted by Verona, but it would be possible to box them.
Without any native support, it is possible to create a Verona generic class that holds an external reference to a C structure of the type of the generic parameter.

Nested structures in C are exposed as objects with `embed` fields in Verona if they are named.
Anonymous structure or union members in C simply add their members directly into the class.

Unions in C are *not* mapped to union types.
There is no way of implementing a `match` expression on a `C` union.
C unions are not type safe and some code depends on writing to one union field and reading from another.
In this way, they are closer to a Verona object type that provides set and get methods of different types and uses internal storage that allows it to materialise a value of the requested type from whatever was set.

Clang will generate a wrapper function for accessing each field.
In the MVP implementation, every field access will generate an RPC.
Most field accesses can be trivially inlined into the caller on the Verona side of the boundary and so by 1.0 the overhead of accessing a field in a C structure should be negligible.

All built-in C types (for example, `int` and `long`) are surfaced as type aliases on the class that is exposed for the library.
A single program may include sandboxes with 32- and 64-bit ABIs for foreign code and so the size of primitive types must be defined per-sandbox-type and, for convenience, exposed per sandboxed-library-instance.

C structures are always exposed in Verona as pointers.
A by-value copy of a C structure can be accomplished with a call to `memcpy`.

### C functions

C functions are defined as methods on the class that is exposed for the library.
They have no `Self` parameter.
All arguments are either `readonly` (if `const` in the C source) or `mut` otherwise.
All types for C functions are types that are exposed for the specific sandbox object.

C function pointers may be used to provide callbacks that invoke Verona code.

### C globals

C globals are defined as fields on the class that is exposed for the library.
`const` C globals are surfaced as `imm` fields, all others as are `mut` in the region defined by an instance of the library.

Mapping C++ constructs to Verona
--------------------------------

Two things are explicitly out of scope from Verona's C++ interop layer:

 - Subclassing a C++ class in Verona.
 - Subclassing a Verona class in C++.

It *is* possible to invoke callbacks from C++ that invoke Verona code and so it is possible to create, on the C++ side of the interop layer, a C++ class that subclasses another C++ class and invokes Verona callbacks for all (or some) of its methods.
Most of the rest of C++ should be possible to expose as Verona concepts.

### Passing values between Verona and C/C++

We treat machine-word types and pointers differently in the interop layer.
The C/C++ primitive types are all exposed as aliases of one of the Verona `Builtin` types (for example, a C `long` in a specific sandbox configuration would be an alias for either `Builtin.I32` or `Builtin.I64`).
These types are passed through the interface layer as simple values.
All other types are represented as pointers.

When a Verona pointer is passed to a C/C++ function, it appears as a `void*`.
In most configurations, this is not going to be a pointer to the Verona object it will instead be a token that identifies an index into a table of Verona objects that have been passed to a given sandbox and which can be used to look up a Verona object if passed back.
In a CHERI world, this will be a (tamper-proof) sealed capability to a Verona object.
This means that it is possible to create, for example, a `std::vector<void*>` that refers to Verona objects, but it is not possible to do anything with these pointers in the C/C++ code other than pass them back to Verona.
Because Verona objects are in a different region to C++ objects, passing a Verona object back from a sandbox will provide an external reference, which must be converted to a normal pointer by presenting the region that contains the Verona object before it can be used.

Pointers to C/C++ objects returned to Verona code are object pointers referring to concrete types.
These are all within the region representing the sandbox and so they can be stored on the stack directly in Verona but can be stored on the heap only as external references.
This means that, for example, a heap-allocated `Array[SomeCXXType & mut]` is not allowed, though `Array[SomeCXXType & ext]` is, though the objects cannot be accessed without presenting the sandboxed region.


### Casts

C++ casts are exposed as generic functions on the class that represents the library.
Any C library has a `$cast[T,U](U) : T` function defined, which takes any C built-in type or pointer type and returns any other.
For C++ codebases, this performs a 'C-style cast', including invoking user-defined conversion operators.
Any C++ library will also expose `static_cast[T,U](U) : T` and `reinterpret_cast[T,U](U) : T` functions and, if compiled with RTTI enabled, a `dynamic_cast[T,U](U) : T` function.
These can all be used with any combination of types for which their C/C++ analogues are defined and with the same semantics.
The Verona syntax implicitly applies to pointers when given any value except a C built-in type.
C built-in types are passed as values to the cast functions unless wrapped in `Ref` cells.
For example:

```verona
// Equivalent to auto *x = new SomeStruct();
var x = C.SomeStruct();
// Equivalent to auto *y = static_cast<SomeOtherStruct*>(x);
var y = C.static_cast[C.SomeOtherStruct](x);
var i : C.int = 12;
// Equivalent to f = static_cast<float>(i);
var f = C.static_cast[C.float](i);
// Equivalent to float *f_type_punned = reinterpret_cast<float*>(&i);
var f_type_punned : C.float = C.reinterpret_cast[Ref[C.float], Ref[C.int]](i);
```

Note that the last line of this is explicitly violating type safety.
All of these operations are restricted to values that are within a sandboxed foreign region where type safety is explicitly not guaranteed.

C++ namespaces
--------------

C++ namespaces are singleton nested classes within the class exposed for the library.
Everything within the namespace is exposed within this context in the same way that it would be if it were declared in the top-level class.
For example:

```c++
// In libfoo
namespace SomeStuff
{
  int x;
};
```

Imported into Verona looks like this:

```verona
import "libfoo" as F;

var f = F();
var x = f.SomeStuff.x : F.int;
```

### Classes

C++ classes have a number of aspects that make them more complex than C structures.

*Visibility specifiers* are respected by Verona code and there is no way to make a Verona class or function a `friend` of a C++ class.
If you want to access a `private` field of a C++ object then this should be done by exposing a `friend` function on the C++ side of the interop layer.

Inheritance is represented in Verona as each subclass defining a new concrete type.
As with normal Verona code, there is *no* implicit conversion from a pointer to a C++ class to a pointer to any of its superclasses, C++ structures passed by pointer to C++ functions must be explicitly converted.
This is one of the places where a programmer writing a wrapper library is likely to want to write some code to simplify the interface (this is not needed for correctness, only for the code to be user friendly).

Verona's unified call syntax provides a mechanism for disambiguating fields in C++ classes that are in both a base class and a subclass.
For example, consider this C++ code:

```c++
struct Super
{
  void x();
};
struct Sub : public Super
{
  void x();
};
```

In Verona, given an instance of `Sub`, you could call `Super::x()` like this:

```verona
var s = CXXLib.Sub();
CXXLib.Super.x(CXXLib.static_cast[CXXLib.Sub](s));
// The following two are equivalent 
s.x();
CXXLib.Sub.x(s);
```

### Templates

C++ templates are exposed as Verona generics.
Unlike Verona generics, C++ templates can take values as type parameters.
For the MVP, we should not support C++20 object literals and so support only types and numeric types as template parameters.
This is sufficient for existing rich C++ APIs and requires approximately the same parsing logic as is already needed for Verona arrays.

Unlike Verona generics, a C++ template may parse but not be possible to instantiate with some of the types that it permits as arguments.
For example, `std::vector<T>` requires that `T` is copy or move constructable but the only constraint on `T` is that it is a type.
To avoid propagating this problem into Verona code, we require that all template arguments are concrete types *before* reification.
A Verona generic can wrap a C++ template by providing a closed set of permitted values for the template instantiation.
If any of these fail, the generic will not be valid Verona code, if all of them succeed then the generic is valid for all possible instantiations.

### Overloaded operators

Verona allows overloading most of the operators that C++ provides and so these can be trivially mapped just like any other methods.
The main exceptions relate to assignment, though Verona's `apply` sugar provides similar functionality.
Classes from C++ need to opt out of `create` and `apply` sugar because these reflect patterns that don't exist in C++ codebases and so it's possible that apply sugar could be applied for C++ assignment operators.

### Exceptions

**This doesn't make me very happy, better suggestions welcome**
Every sandboxed function may throw a `SandboxError` exception if the sandbox has failed and so all invocations of sandboxed code must be wrapped in a `try` block.

C++ has unchecked exceptions and so any function not explicitly declared as `noexcept` may throw any type.
Verona, in contrast, has only checked exceptions.
This provides us with a way of identifying the expected exceptions:
Any exception of a type from a particular sandboxed library that the code invoking the foreign code either explicitly catches or advertises that it may throw is assumed to potentially originate in the sandbox.
Code inside the sandbox will catch *all* exceptions and either expose them to the caller or report an error that will trigger a `SandboxError`.

Implementation
--------------

C/C++ are among the most difficult cases for interoperability, for two reasons:

 - They lack of a clear module system or interface format. 
   C/C++ 'interfaces' are header files.  
   These are just text concatenation and can contain arbitrary C/C++ code.
   They have to be extracted by compiling the code and cannot be recreated with 100% fidelity by anything that is not a complete C/C++ preprocessor, parser, and semantic analysis toolchain.
 - ABI complexity. 
   The standard C ABI is not too hard to handle for any given platform (except bitfields) but there are a load of non-standard GCC and MSVC extensions that give fine-grained control over structure layout.
   Beyond that, C++ is even harder.
   Things like diamond inheritance (handled differently in Itanium / MSVC ABIs), RTTI (handled completely differently in MSVC vs Itanium, subtly differently in Itanium vs Aarch32 vs Fuchsia), exception ABIs (totally different between Win32, Win64, and Itanium, with some significant differences between ARM and x86 Windows) add up to a very large amount of ABI logic that differs between platforms and architectures.
   Getting these right is really, really hard in anything that isn't a complete C/C++ compiler.

To address this, the proposed implementation strategy for C/C++ interoperability is to deeply embed clang.
We are already depending on LLVM (and MLIR) for our compiler infrastructure and so embedding clang is not a significant addition.
We would pass the header that defines the interface and the flags that describe how to parse it to clang and generate an AST.
Any lookup of symbols within the imported class's namespace would inspect this AST to find equivalent types.

We want to avoid having to deal with the C/C++ ABI details in the Verona compiler.
A suggestion from Chris Lattner provides a path to avoiding that by having clang synthesise wrapper functions.

For example, consider some Verona code that wants to call `llvm::IRBuilder<>::CreateRetVoid`.
This is a method in a templated class, so the template will need to be instantiated before it can be called and the caller then needs to be aware of C++ ABI details.
This, like every other problem in computer science, can be addressed by a level of indirection, using clang to synthesise a function roughly equivalent to this :

```c++
struct irbuilder_createretvoid_args
{
  llvm::IRBuilder<> *b;
  llvm::ReturnInst *ret;
};

__attribute__((used,always_inline))
extern "C" void irbuilder_createretvoid(irbuilder_createretvoid_args *args)
{
        args->ret = args->b->CreateRetVoid();
}
```

This takes a single pointer as an argument (and so has a trivial calling convention in LLVM IR).
Clang emits a simple function for this:

```llvm
; Function Attrs: alwaysinline uwtable
define dso_local void @irbuilder_createretvoid(%struct.irbuilder_createretvoid_args* nocapture %0) #0 {
  %2 = bitcast %struct.irbuilder_createretvoid_args* %0 to %"class.llvm::IRBuilderBase"**
  %3 = load %"class.llvm::IRBuilderBase"*, %"class.llvm::IRBuilderBase"** %2, align 8, !tbaa !2
  %4 = tail call %"class.llvm::ReturnInst"* @_ZN4llvm13IRBuilderBase13CreateRetVoidEv(%"class.llvm::IRBuilderBase"* %3)
  %5 = getelementptr inbounds %struct.irbuilder_createretvoid_args, %struct.irbuilder_createretvoid_args* %0, i64 0,    i32 1
  store %"class.llvm::ReturnInst"* %4, %"class.llvm::ReturnInst"** %5, align 8, !tbaa !7
  ret void
}
```

Creating this forced clang to instantiate a lot of templated functions (17 at time of writing), with almost all of them end up being inlined at higher optimisation levels.
Fortunately, [Hal Finkel's C++ JIT prototype](https://github.com/hfinkel/llvm-project-cxxjit) demonstrated that this is possible without changes to the clang AST.

This generated function is then trivial for the Verona compiler to call (all details of the C/C++ ABI are abstracted) and can then be inlined during the optimisation phase if desirable.
More interestingly, LLVM will inline the `IRBuilder<>::CreateRetVoid` call into this function.
The Verona compiler can then examine the result and determine if it is safe to inline at the caller without breaking the sandboxing invariants.
For example, a function that ends up flattened can be made safe simply by enforcing the sandbox bounds checks on every LVM IR load and store instruction.
Functions that do a lot of work, in contrast, may require a full domain switch.

Exactly the same technique can be applied to field accessors.
Getting and setting a field in a C or C++ POD `struct` is easy in the common cases, but may involve some complex ABI interactions for features such as bitfields in structures where some fields have custom alignment.
To access these fields, the Verona compiler would simply make clang synthesize a 'get' and 'set' function for each field.
This can then be inlined (with correct bounds checks applied).

A similar technique applies to memory allocation.
Without optimisation, any 'on stack' value types that refer to a sandbox's heap should be allocated in that heap and have their lifetimes tracked with the owning stack frame.
Ideally, we'd promote these to the stack.
This is possible if every function that is called on them can be safely inlined.
For example, consider `std::shared_ptr` or `std::lock_guard`.
These are almost stateless in their common implementation and refer to state stored on the heap.
These would be safe to allocate on the stack (or even in a register: the storage is a single pointer).
Methods may invoke calls into the sandbox, but those calls would not take the address of the on-stack object.  

There are lots of open questions in terms of exposing C++ syntax (for example, operator overloading).

Exporting Verona code to other languages
----------------------------------------

**TODO**
