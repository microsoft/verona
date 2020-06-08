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
For example, we may import LLVM in something like this (straw-man syntax):

```verona
class LLVM
{
	[#native, #interface_language="c++"]
	imports library("libLLVM.so", "llvm.h");
}
```

This includes two attributes that tell the compiler the format of the library (native code) and how to parse the header (which, in this example, would have to be something provided by the user that included the desired headers).
Once C++ module support is more widespread, module imports can take the place of header inclusions.

This instructs the compiler to expose the C++ namespace and class structure the `LLVM` Verona class.
C++ namespaces and classes will appear as nested classes within the Verona `LLVM` class.
For example, `LLVM.llvm.SmallVector[LLVM.int, 12]` corresponds to `llvm::SmallVector<int, 12>`.

Instantiating this class would require the user to specify a sandboxing policy, for example (again, straw-man syntax):

```verona
var llvm = LLVM.Create(PROCESS);
```

This would create a new process that loaded the LLVM library and had RPC invocations for any of the LLVM functions that this program called.
The `llvm` object in this example is a new isolated region.
The programmer can then construct LLVM types within that region.

Implementation
--------------

C/C++ are among the most difficult cases for interoperability, for two reasons:

 - They lack of a clear module system or interface format. 
   C/C++ 'interfaces' are header files.  
   These are just text concatenation and can contain arbitrary C/C++ code.
   They have to be extracted by compiling the code and cannot be recreated with 100% fidelity by anything that is not a complete C/C++ preprocess, parser, and semantic analysis toolchain.
 - ABI complexity. 
   The standard C ABI is not too hard to handle for any given platform (except bitfields) but there are a load of non-standard GCC and MSVC extensions that give fine-grained control over structure layout.
   Beyond that, C++ is even harder.
   Things like diamond inheritance (handled differently in Itanium / MSVC ABIs), RTTI (handled completely differently in MSVC vs Itanium, subtly differently in Itanium vs Aarch32 vs Fuchsia), exception ABIs (totally different between Win32, Win64, and Itanium, with some significant differences between ARM and x86 Windows) add up to a very large amount of ABI logic that differs between platforms and architectures.
   Getting these right is really, really hard in anything that isn't a complete C/C++ compiler.

To address this, the proposed implementation strategy for C/C++ interoperability is to deeply embed clang.
We are already depending on LLVM (and MLIR) for our compiler infrastructure and so embedding clang is not a significant addition.
In the above example, the we would pass the `llvm.h` header to clang and generate an AST.
Any lookup of symbols within the `LLVM` class's namespace would inspect this AST to find equivalent types.

We want to avoid having to deal with the C/C++ ABI details in the Verona compiler.
A suggestion from Chris Lattner provides a path to avoiding that by having clang synthesise wrapper functions.

For example, consider some Verona code that wants to call `llvm::IRBuilder<>::CreateRetVoid`.
This is a method in a templated class, so the template will need to be instantiated before it can be called and the caller then needs to be aware of C++ ABI details.
This, like every other problem in computer science, can be addressed by a level of indirection, using clang to synthesise a function roughly equivalent to this :

```c++
__attribute__((used,always_inline))
extern "C" llvm::ReturnInst *irbuilder_createretvoid(llvm::IRBuilder<> *b)
{
        return b->CreateRetVoid();
}
```

This takes and returns all values as pointers and so the LLVM lowering is predictable as a function that takes and returns only pointers.
For complex return values, the caller will need to allocate space and pass a pointer to the return.
Clang generates IR somewhat like the following for this function.

```llvm
define dso_local %"class.llvm::ReturnInst"* @irbuilder_createretvoid(%"class.llvm::IRBuilder"*) #0 {
  %2 = alloca %"class.llvm::IRBuilder"*, align 8
  store %"class.llvm::IRBuilder"* %0, %"class.llvm::IRBuilder"** %2, align 8
  %3 = load %"class.llvm::IRBuilder"*, %"class.llvm::IRBuilder"** %2, align 8
  %4 = bitcast %"class.llvm::IRBuilder"* %3 to %"class.llvm::IRBuilderBase"*
  %5 = call %"class.llvm::ReturnInst"* @_ZN4llvm13IRBuilderBase13CreateRetVoidEv(%"class.llvm::IRBuilderBase"* %4)
  ret %"class.llvm::ReturnInst"* %5
}
```

Creating this forced clang to instantiate a lot of templates.
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
