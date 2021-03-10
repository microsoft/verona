# Verona Interoperability / Sandbox

This is an initial attemp at mapping all the requirements that the interoperability layer will have in Verona to use external code written in other languages (like C++, Rust, etc) in a safe manner (sanbdox).

## Usage in Verona

The aim is to have external code in Verona with a native interface that looks and feels like Verona code.

We want to abstract away the entirety of the sandbox functionality, the foreign function dispatch and marshalling, the execution models, etc.

The syntax is still largely undefined, but this is an example of using the C++ `std::vector` class inside Verona as pseudo-Verona code:

```c
// import a module in ./extern/StdVec with a config file exposing the includes and libraries.
// Declares the "StdVec" module with specific language and sandbox declarations
using "extern/StdVec";

// Create a new region (foreign, so parametrised on the sandbox type)
// This syntax is not defined yet, here we assume this is how we get a particular sandbox
// "int" here is relative to "StdVec", which defined all C++ types (from config) as an alias to a Verona type
vec = new StdVec[ProcessSandbox]::std::vector[int]();

// Use the object as if it was native
// "42" is "int", since it comes from the template parameter above
vec.push_back(int(42))

// The result will be used by a native Verona function directly, as it's an alias to a known Verona type
// "0" is "size_t" as defined by std::vector, which alias a known Verona type
print(vec.at(size_t(0)))
```

## Overall architecture

From the point of view of the Verona program, `StdVec` is a module just like any other in Verona. Language, sandbox and platform declarations (types, behaviour) are done in Verona declarations.

The "extern/StdVec" directory has a configuration file, that will tell us all we need to know about the language, headers, libraries, etc. That's what drives the correct declarations in the "StdVec" module.

Not all foreign language functionality will be available to the Verona code, especially in the beginning.

Programmers are expected to create thin layers of foreign code to wrap functionality that can't be used in Verona (like what "extern C" does in C++).

Different sandbox technologies and platforms have different binary representation for the same types. We can use type aliases when constructing the modules so that `int` alias the right Verona type (ex. `I32` or `I64`).

Programmers are also expected to wrap existing Verona classes (using fixed-width Verona types) with explicit types (and casts) that can work with any sandbox technology.

Each external language needs explicit support in the Verona compiler to be parsed at compile time and generate the correct stubs for the interoperability layer to correctly implement the remote procedure call (RPC).

Each execution model (child process, web assembly, etc) will need explicit support for deploying the functionality at run time.

But the overall architecture will be the same.

1. The code will be in pure Verona language. There will be a compiler generated translation layer between Verona syntax and any supported language.
2. The `parser` will generate Verona AST, creating a module for each foreign sandbox and adding all functions and types called from Verona to it as Verona nodes.
3. Verona AST will be lowered to `1IR` as is.
4. The `type checker` will query the foreign layer for the existence and validity of the corresponding Verona types.
5. `1IR` is lowered to `2IR` (`MLIR`) as is.
6. The `codegen` layer will take concrete declarations and generate `LLVM IR` for each foreign function and wrappers for each module in a new object file, as well as `MLIR` code for each Verona function and wrapper.
7. `MLIR` will be lowered to `LLVM IR` as is.
8. LLVM will generate machine code for the main code (Verona), for each foreign module in a separate object, and will statically link all auto-generated sandbox objects to the main object.
9. Any dynamic libraries specified in the module configuration will be linked at run time.

Each step below in more details.

### Parser

The Verona parser doens't know about any foreign language. The source code is in Verona, so Verona AST nodes are created.

The Verona AST allows for unknown types, so we can't yet query concrete foreign types. It also doens't have all concrete functions (from templates), so we also can't query about specific functions.

Verona modules are directories in the file system, with source files to be parsed and added as a type. Foreign modules can't be parsed by the Verona parser, so we keep a configuration file with the details, in addition to any foreign source files.

For each foreign module (language/sandbox/region), the parser creates a Verona module (with associated configuration file).

For each class and function call made to a foreign module, the parser creates a declaration on the respective module, keeping the source location of the call to use on error messages.

The end result is an AST that has foreign nodes as Verona nodes on Verona modules without the actual implementation.

For all purposes, this is a plain Verona AST with calls to functions that have been declared, but not defined.

Definition (and implementation in object code) will be done later by the compiler once we know all functions that have survived the initial round of cleanups (ex.  reachability analysis).

### Type checker

During type inference and reification, the type checker will have to make sure the foreign types are valid, just like Verona types. However, the Verona compiler doesn't know anything about foreign types.

On sandbox modules, language types are defined as aliases to known Verona types. This is constructed by the compiler with knowledge about the language, the sandbox technology and the target platform.

At this stage, a foreign translation layer will have to be created, with the main functionality to:
* Parse the foreign source files in the module directory and create a foreign AST representation.
* Verify that types exist and are valid (including template types such as `std::vector<int>`).
* Verify that specific function signatures (name, arity, types) exist in the foreign module (ex. `void std::vector<int>::push_back(int)`).

With the foreign layer in hand, the type checker can make sure all inferred types are valid in the foreign language (correct and available).

As a first implementation, we won't allow Verona template arguments in foreign code template arguments, to avoid combinatorial explosion of type checks.

But with all types concrete, the compiler can now check each function declaration for their foreign code implementations.

For example, from the Verona perspective, `std::vector` is a generic class that takes a `T` constrained by a union type (`int`, `float`, `FooBar`, ...).

Built-in types are added to the union by default, user-declared types are added for each case.

You can check if a Verona union type is a subtype of this union type by checking whether the template can be instantiated with each Verona type (fast fail if the Verona type is not a C/C++ type).

Any Verona type that is less constrained than a union type over a set of concrete C/C++ types is not a subtype of the constrained type for `T`.

With the source location propagated from the module, which came from the original Verona call line, we can inform programmers what are the actual Verona and foreign types from that specific line in case of errors.

### Code generation

With all concrete types and calls validated by both Verona and foreign checks, we now know that we can implement all foreign calls and type conversions in the compiler.

The compiler will generate code in two ways:
* All Verona code referring to a concrete call and argument marshalling will be lowered by the Verona compiler in the main object file.
* All foreign code referring to specialisation and argument marshalling will be lowered by the foreign layer compiler in a separate object.

So, for each sandbox region, the compiler:
1. Generates calls to the runtime to setup the region and sandbox.
2. Creates a dispatcher for the RPC mechanism, with index and a buffer for arguments and return values.
3. Creates a foreign compile unit (via foreign interface) with required boilerplate for the language.

The foreign sources and headers have already being compiled into AST at type checking, so we still have all the information we need to lower the concrete implementations.

Within a particular sandbox, for each function called into the foreign module:
1. If the call is a foreign template specialisation, the language driver implements the actual specialised function on the foreign object.
2. The foreign module generates a marshalling function from buffer to foreign types and calls the final function, returning an auto-increment index.
3. The foreign module adds that function to a dispatch table and return the index to the compiler.
4. The compiler implements the Verona module function version with Verona arguments and return values, marshalling them into a memory buffer and calling the dispatcher with that index.
6. The appropriate object files are created and later linked with the main executable.
7. Dynamic libraries will be linked at run time.

The compiler is responsible for emitting errors that inform the programmer what type/function cannot be represented and potentially why.

The programmer is responsible for implementing a wrapper function that abstracts away the complexity of the language in a public function that Verona can call.

The language design is responsible for providing a way to allow portable code to be written in Verona that can run in different sandbox technologies, from different languages and on different target platforms.

Language support will be always evolving. This is expected to be a balance between "best effort" and "appropriate support".

### Run time

When the main process starts, it will dynamically link all foreign libraries that were mentioned in the foreign modules configurations.

All auto-generated foreign code has already been statically linked to the main object at compile time.

When the sandbox is initialised, its allocator sets up a new isolated heap that is used for all allocations in the sandbox (including Verona foreign objects that belong to that region).

Verona passes the heap to `snmalloc` on the sandbox side as its memory to manage. All calls to allocate and free are local but done on that region only.

Other sandbox code calls may be done directly (if safe, like reading from an already open file descriptor) or redirected to the sandbox driver (upcall) to decide what to do.

For each sandbox function call:
1. The first call is made to the Verona module's wrapper, which will marshall the arguments and call the dispatcher with a compile-time constant index.
2. The dispatcher will call the foreign wrapper to marshall back into foreign types and call the actual function.
3. The return value, if any, is marshalled back into the buffer.
4. The dispatcher returns the buffer to the Verona module's wrapper.
5. The wrapper extracts the return value and return to the caller.

This sequence is entirely defined at compile time: How to marshall arguments and return values on either end, what is the constant index and the actual function to call.

## Use Case: C/C++ interoperability

For both C and C++, we use `clang` to parse the header file to know what types are declared, functions are exported, etc.

This allows us to know which of the Verona code lines calling foreign code are correct, and to generate the correct marshalling routines and, for C++, instantiating the correct template implementations based on the types used in Verona.

The C++ interoperability layer parses the file and generates the clang AST.

We then use the Verona code function names and types to create a function declaration (context, name, argument types) and, if valid, we add that function to the dispatcher, giving it its own (auto-incremented) index.

Additional code needs to be generated to marshall the Verona arguments and return values into a buffer, so that the dispatcher can be generic, only taking an index and a memory buffer.

That code is part of the sandbox library, as a template function on the arguments and return values, that is generated by the compiler upon encountering each specific foreign function.

For C++, specialisation code will be created (via AST construction and LLVM code generation) if the types involved are parametrised.

Note that, for the first implementation, we won't allow Verona generic parameters to be passed to template instantiations.

This means that all possible template instantiations that a Verona generic may use must exist before reification, which means we have the same guarantees for Verona generics that use C++ templates as we do for other Verona generics (i.e.  they either type check and work for all instantiations or they don't type check).

### Child process execution environment

Our initial implementation is to separate the sandbox execution by creating a child process. This is not strictly speaking a sandbox, but gives us an easy way to test the rest of the framework without much additional work.

Calls will be forwarded across the parent/child barrier via the dispatcher, by means of a function pointer table, where the position is the index in the code generation stage above, and the buffer is the contents of the arguments and a place for the return value.

Argument decomposition will be done by the sandbox code that will call the actual function with the actual arguments and, if there is a return value, set it on the right place of the buffer and return.

The main driver in the child process will be a loop taking calls into the sandbox, calling them and returning the buffer to the dispatcher.

### Memory Management

To avoid the sandbox from requesting memory in the wrong places (accidentally or maliciously), we use `snmalloc`'s ability to allocate memory in slabs that aren't managed by themselves.

The allocator in the Verona side manages the slabs and pass their ownership to the sandbox allocator, which then allocates directly on their reserved heap.

This is part of the sandbox library's functionality, which is passed along on the creation of the sandbox.

Because that heap is in a Verona sandbox region, it is safe to assume there are no race conditions introduced by the compiler onto the external library, due to the use of cowns to execute the foreign code.

## Step-by-step example

This is the description of a mock implementation of the child process sandbox running the C++ snippet in the beginning of this document.

In this example, we'll go line by line and describe what the compiler and the generated code will do for each line. This is a orthogonal view as described above (per line, not per compiler stage).

```c
using "extern/StdVec";
```

The compiler will:
1. Create a Verona module called `StdVec` for the foreign module.
2. Read the module description file and discover that it's a C++ module.
3. Call the C++ driver (clang wrapper) to parse the include file (and all its includes) and keep as a C++ AST.
4. Expose an interface to query types and functions.

No discernible differences at run time.

```c
vec = new StdVec[ProcessSandbox]::std::vector[int]();
```

The compiler will:
1. Create a new region using `ProcessSandbox`:
   1. Recognise the type as coming from a foreign module of a specific sandbox type (syntax pending).
   2. Lower code to create a region with a sandbox allocator (RT calls).
   3. Lower code to create the sandbox memory area (RT allocator calls), taking the maximum heap size from the sandbox type.
   4. Create a wrapper object for function dispatch and general sandbox utilities.
2. Setup the constructor calls:
   1. Recognise that is a class constructor and declare the class with the right number of template parameters.
   2. Assemble the foreign call signature: `std::vector<int> std::vector<int>::vector<int>()`
   3. Instantiate the signature as a template specialisation in the AST, using the query interface to validate the code and get an actual implementation.
   4. Create a constructor in Verona called `create(): vector[int]` with `int` being the specific sandbox's own alias.
   6. Generate the wrapper function from memory buffer to return value.
   7. Add that function to the sandbox dispatcher's table and associate the index with this particular specialisation, increment the index.
   9. Bind the result of the Verona call to the variable `vec` in the new region.

At run time:
1. Initialise the region:
   1. A Cown is created with a specific sandbox allocator.
   2. Memory allocated by the parent allocator, used by the sandbox allocator.
   3. Any pending libraries (shared objects?) are loaded.
   4. The Cown is pushed to the queue.
2. Call the constructor:
   1. `StdVec::std::vector[int]::create()` will be called, which calls the dispatcher for the region with index `0` and no args.
   2. The sandbox dispatcher calls the function pointer, which is a template object that has the actual function pointer plus the templated argument/return handling.
   3. The dispatcher returns, with the return value in the memory buffer.
   4. The Verona code reconstructs a Verona type from the return value in the buffer and returns from `StdVec::std::vector[int]` as an object.
   5. Stores the return value in the memory pointed by the variable `vec`.

```c
obj.push_back(int(42))
```

The compiler will:
1. Validate the type of `42` (something like `C::int`, alias to something like `I32`).
2. Assemble the foreign call signature: `void std::vector<int>::push_back(int)`
3. Same as above, creates a new Verona function in the `StdVec` class that will marshall arguments and return value and call the dispatcher.
4. The Verona signature would be something like `StdVec::std::vector[int]::push_back(int)`.
5. Same as above, lowers C++ implementations of the functions on the foreign object file and append the function pointer to the dispatcher's table.
6. Associate the Verona call to the newly created in `StdVec` class.

At run time:
1. `StdVec::std::obj.push_back(int)` will be called, which calls the dispatcher for the region with index `1` and marshalled `int(42)`, for example, a 32-bit signed integer.
2. Sandbox dispatcher calls the function, which is from a template object that has the actual function pointer plus the templated argument/return handling.
3. There is no return value, so `obj.push_back(int)` just returns.

```c
print(obj.at(size_t(0)))
```

The compiler will:
1. Validate the type of `0` (something like `C::size_t`) and the return type (`C::int`).
2. Try to call a function on the foreign module.
3. Assemble the foreign call signature: `int& std::vector<int>::at(size_t)`
4. Same as above, creates a new Verona function in the `StdVec` class that will marshall arguments and return value and call the dispatcher.
5. The Verona signature would be something like `StdVec::std::vector[int]::at(size_t) : int`.
6. Same as above, lowers C++ implementations of the functions on the foreign object file and append the function pointer to the dispatcher's table.
7. Associate the Verona call to the newly created in `StdVec` class.
8. Uses the returned value to pass as an argument to the `S32::cast()` function.
9. Implement the cast function (via the specific sandbox library, from the specific sandbox type).
10. Passes the cast value to the `print` function.

At run time:
1. Calls `obj.at(size_t)`, which calls the dispatcher for the region with index `2` and marshalled `size_t(0)`, for example, an unsigned 64-bit integer.
2. Sandbox dispatcher calls the function, which is from a template object that has the actual function pointer plus the templated argument/return handling.
3. Dispatcher returns, with the return value in the memory buffer.
4. The Verona code reconstructs a Verona type from the return value in the buffer and returns from `obj.at(size_t)` as `int(42)`.
5. Calls the function `print` with the argument from the return value after a potential cast.

### Discussion

Note that the functions are almost identical, mainly:
* They all have a Verona definition (types, functions) which is what other Verona code will see, and is what converts types and call the dispatcher.
* They all find the dispatcher based on the region (which knows function indices and have the right buffer handling wrappers).
* They all call the dispatcher with a (run-time) constant index, defined at compile time, and the memory buffer created on the fly.
* They all compute the signature and instantiate the code. However, if the function has been called already, it already has a Verona implementation, so we just call it.
* They all end up in the dispatcher as an index and a buffer, and it's up to marshalling code to make sure the shape of that memory region is compatible from both sides, Verona and the external language.
* All foreign calls go through the dispatcher.

We generate marshalling code on the fly to allow for better compiler optimisations.
