# Verona Interoperability / Sandbox

This is an initial attemp at mapping all the requirements that the interoperability layer will have in Verona to use external code written in other languages (like C++, Rust, etc) in a safe manner (sanbdox).

## Usage in Verona

The aim is to have external code in Verona with a native interface that looks and feels like Verona code.

We want to abstract away the entirety of the sandbox functionality, the foreign function dispatch and serialisations, the execution models, etc.

The syntax is still largely undefined, but this is an example of using the C++ `std::vector` class inside Verona as pseudo-Verona code:

```c
// include/import the c++ code as a Verona module
type StdVec = "c++::vector";

// Create a new region (we know it's foreign, so it's a sandbox)
reg = new StdVec;

// Use Verona syntax to initialise a C++ vector
obj = new std::vector[int] in reg

// Use the object as if it was native
obj.push_back(42)

// The result will be used by a native Verona function
print(obj.at(0))
```

The only hint that `obj` is a foreign object is in the region type being a sandbox.

Not all foreign functionality will be available to the Verona code, especially in the beginning, so programmers are expected to create thin layers of foreign code to wrap functionality that can't be used in Verona (like what "extern C" does in C++).

The syntax for foreign language code in Verona hasn't been defined yet, so we don't discuss it here.

The pseudo-code in here is only a hint to start the discussion on the implementation.

The objective of this document is to describe:

* how is this going to interoperate with Verona and the target language
* how the remote procedure calls will be dispatched and the arguments serialised
* how the runtime will separate the memory from the main process and sanitise the execution (system calls, file descriptors, ioctl, etc)

## Overall architecture

Each external language needs explicit support in the Verona compiler to be parsed at compile time and generate the correct stubs for the interoperability layer to correctly implement the remote procedure call (RPC).

Each execution model (child process, web assembly, etc) will need explicit support for deploying the functionality at run time, hopefully not individually bound to the source language (N-to-M).

But the overall architecture will be the same.

### Parser

1. Upon finding a module import that has foreign origin, the appropriate foreign language parser (clang for C++) will be asked to parse the file and generate its own AST.
2. A Verona module will be created, with all foreign builtin types.
3. When the Verona compiler asks for a particular type or function in the foreign module, the language parser will convert the signature of its own AST into Verona AST and add that node in its own module, then pass that to the Verona AST node.

This allows lazy evaluation of what could be very large number of includes, with code that can't be translated into Verona.

The end result is an AST that has foreign nodes as Verona nodes on Verona modules without the actual implementation.

For all purposes, this is a plain Verona AST with calls to functions that have been declared, but not defined.

Definition (and implementation in object code) will be done later by the compiler once we know all functions that have survived the initial round of cleanups (ex.  reachability analysis).

### Compiler

1. For each sandbox region, the compiler generates calls to the runtime to setup the region, including its allocators.
2. Upon finding a foreign call, the compiler will ask the language driver for the foreign AST node, to implement the actual call.
3. The driver will generate the conversion between Verona and native types, including structure layout handling per architecture, serialise the arguments and return values into a buffer and call the dispatcher below.
4. A dispatcher will be created to call those functions with serialised arguments by an index that is associated when they're created.
5. A bi-directional RPC layer is created between the Verona code and the sandbox, where calls to external functions go one way, and calls to allocation, system calls, etc. go the other way.
6. The appropriate object files are created and linked with the main executable.

Not all functions will be representable in Verona, not all structure layouts will be easily known at the Verona level (special hardware ABI), so errors may still occur here.

The compiler is responsible for emitting errors that inform the programmer what type/function cannot be represented and potentially why.

The programmer is responsible for implementing a wrapper function that abstracts away the complexity of the language in a public function that Verona can call.

Language support will be always evolving. This is expected to be a balance between "best effort" and "appropriate support".

### Run time

1. When the sandbox is initialised, its allocator sets up a new isolated heap that is used for all allocations in the sandbox (including Verona foreign objects that belong to that region).
2. When the foreign code makes calls to allocate memory or run potentially unsafe operations (ex: opening files, system calls, etc.), the RPC layer may check them and forward if explicitly marked as an upcall or if it's a direct sandbox functionality (webasm, CHERI).
3. When Verona code makes calls to the foreign library, the compiler-generated marshalling code will transform the arguments and return value into a memory buffer and call the dispatcher with the appropriate index and buffer.
4. The dispatcher will call a templated wrapper that unpacks the arguments and calls the actual function.
5. The return value, if any, is desserialised and bound to the Verona variable or used as an argument in a Verona call.

## C/C++ interoperability

For both C and C++, we use `clang` to parse the header file to know what types are declared, functions are exported, etc.

This allows us to know which of the Verona code lines calling foreign code are correct, and generate the correct serialisation routines and, for C++, instantiating the correct template implementations based on the types used in Verona.

The C++ interoperability layer parses the file and generates the foreign AST.  We then use the Verona code function names and types to create a function declaration (context, name, argument types) and, if valid, we add that function to the dispatcher, giving it its own (auto-incremented) index.

Additional code needs to be generated to serialise the Verona arguments and return values into a buffer, so that the dispatcher can be generic, only taking an index and a memory buffer.

For C++, further code will be created (via AST construction and LLVM code generation) if the types involved are parametrised.

All template arguments are taken into consideration and a specialisation is defined and generated in a wrapper layer.

Note that, for the first implementation, we won't allow Verona generic parameters to be passed to template instantiations.

This means that all possible template instantiations that a Verona generic may use must exist before reification, which means we have the same guarantees for Verona generics that use C++ templates as we do for other Verona generics (i.e.  they either type check and work for all instantiations or they don't type check).

## Child process execution environment

Our initial implementation is to separate the sandbox execution by creating a child process.

This is not strictly speaking a sandbox, but gives us an easy way to test the rest of the framework without much additional work.

The sandbox creation forks the process with a special sandbox library that will forward all calls to the parent process, which will control where the memory actually lives (see below). A local heap is created per sandbox for the child to use in its allocations.

Calls will be forwarded across the parent/child barrier via the dispatcher, by means of a function pointer table, where the position is the index in the code generation stage above, and the buffer is the contents of the arguments and a place for the return value.

Argument decomposition will be done by the sandbox code that will call the actual function with the actual arguments and, if there is a return value, set it on the right place of the buffer and return.

### Memory Management

To avoid the sandbox from requesting memory in the wrong places (accidentally or maliciously), we use snmalloc's ability to allocate memory in slabs that aren't managed by themselves.

The allocator in the Verona side manages the slabs and pass ownership of them to the sandbox allocator, which then allocates directly on their reserved heap.

This is part of the sandbox library's functionality, which is passed along on the creation of the sandbox.

The Verona allocator creates a heap for the sandbox and that's the memory that the sandbox code can use for everything. The sandbox code can also access that memory and it uses to keep the arguments and return values, so that the sandbox can see them.

Because that heap is in a Verona sandbox region, it is safe to assume there are no race conditions introduced by the compiler onto the external library, due to the use of cowns to execute the foreign code.

## Step-by-step example

This is the description of a mock implementation of the child process sandbox running the C++ snippet in the beginning of this document.

We'll go line by line and describe what the compiler and the generated code will do for each of them.

### Compile time

```c
type StdVec = "c++::vector";
```

1. Find the module that the import directive corresponds to.
2. Read the module description file and discover that it's a C++ module.
3. Call the C++ driver (clang wrapper) to parse the include file (and all its includes) and keep as a C++ AST.
4. Create a Verona class called `StdVec` for the foreign module.
5. Expose an interface for the parser to query symbols (types, objects and functions).

```c
reg = new StdVec;
```

1. The parser recognises the type as coming from a specific module. It queries that module for types within that module as they are used.
2. Lower code to create a region with a sandbox allocator (RT calls).
3. Lower code to create the sandbox memory area (RT allocator calls), taking the maximum heap size from the sandbox type.
4. Loads all necessary dynamic libraries specified in the module.
5. Create a wrapper object for function dispatch and general sandbox utilities.
6. The region is bound to the variable `reg`.

```c
obj = new std::vector[int] in reg
```

1. The parser recognises the region and makes sure the object to be created is placed in the same sandbox, after calling the appropriate foreign constructor.
2. Assemble the foreign call signature (namespace, template args): `std::vector<int> std::vector<int>::vector<int>()`
3. Instantiate the signature as a template specialisation in the AST, using the query interface to validate the code and get an actual implementation.
4. If available, create the Verona equivalent inside the `StdVec` class and Lower call to that function. Otherwise emit an error and stop processing.
5. The Verona signature would be something like `create() : StdVec::std::vector[S32]`.
6. Generate the wrapper function from memory buffer to return value.
7. Add that function to the sandbox dispatcher's table and associate the index with this particular specialisation, increment the index.
8. On the recently created `StdVec` Verona function, lower Verona code to call the dispatcher with the index and the buffer and to extract the return value.
9. Bind the result of the Verona call to the variable `obj`.


```c
obj.push_back(42)
```

1. The parser tries to call a function on the foreign module.
2. Assemble the foreign call signature (namespace, template args): `void std::vector<int>::push_back(int)`
3. Same as above, creates a new Verona function in the `StdVec` class that will serialise arguments and return value and call the dispatcher.
4. The Verona signature would be something like `StdVec::std::vector[S32]::push_back(S32)`.
5. Same as above, lowers C++ implementations of the functions on the foreign object file and append the function pointer to the dispatcher's table.
6. Associate the Verona call to the newly created in `StdVec` class.

```c
print(obj.at(0))
```

1. The parser tries to call a function on the foreign module.
2. Assemble the foreign call signature (namespace, template args): `int& std::vector<int>::at(size_t)`
3. Same as above, creates a new Verona function in the `StdVec` class that will serialise arguments and return value and call the dispatcher.
4. The Verona signature would be something like `StdVec::std::vector[S32]::at(U64) : S32`.
5. Same as above, lowers C++ implementations of the functions on the foreign object file and append the function pointer to the dispatcher's table.
6. Associate the Verona call to the newly created in `StdVec` class.
7. Uses the returned value to pass as an argument to the `print` function.

Note that the three functions are almost identical, mainly:
* They all compute the signature and instantiate the code. However, if the function has been called already, it already has a Verona implementation, so we just call it.
* They all end up in the dispatcher as an index and a buffer, and it's up to serialisation and de-serialisation code to make sure the shape of that memory region is compatible from both sides, Verona and the external language.
* All foreign calls are to the dispatcher.

The main differences are:
* How we find the template types by either looking at explicit declarations, such as `std::vector[int]` or by looking at the object it's associated to, ie. `obj`.
* How we only generate serialisation for the things we use (args, rets, both) and don't have a generic wrapper method to do them all. This strategy copes better with inlining and other optimisations.

### Run time

```c
type StdVec = "c++::vector";
```

* This line has no run time consequences on its own.

```c
reg = new StdVec;
```

1. Cown is created with a sandbox allocator.
2. Sandbox memory allocated by the parent allocator.
3. Load any pending libraries (shared objects?).
4. Stores the cown in the variable `reg`.

```c
obj = new std::vector[int] in reg
```

1. Calls `StdVec::std::vector[int]::create()`, which calls the dispatcher for `reg` with index `0` and no args.
2. Sandbox dispatcher calls the function pointer, which is a template object that has the actual function pointer plus the templated argument/return handling.
3. Dispatcher returns, with the return value in the memory buffer.
4. De-serialisation code reconstructs a Verona type from the return value in the buffer and returns from `StdVec::std::vector[int]` as an object.
5. Stores the return value in the variable `obj`.

```c
obj.push_back(42)
```

1. Calls `obj.push_back(int)`, which calls the dispatcher for `reg` with index `1` and serialised `S32(42)`.
2. Sandbox dispatcher calls the function, which is from a template object that has the actual function pointer plus the templated argument/return handling.
3. There is no return value, so `obj.push_back(int)` returns.

```c
print(obj.at(0))
```

1. Calls `obj.at(size_t)`, which calls the dispatcher for `reg` with index `2` and serialised `U64(0)`.
2. Sandbox dispatcher calls the function, which is from a template object that has the actual function pointer plus the templated argument/return handling.
3. Dispatcher returns, with the return value in the memory buffer.
4. De-serialisation code reconstructs a Verona type from the return value in the buffer and returns from `obj.at(size_t)` as `S32(42)`.
5. Calls the function `print` with the argument from the return value.

Again, all functions are largely similar:
* They all have a Verona definition (types, functions) which is what other Verona code will see, and is what converts types and call the dispatcher.
* They all find the dispatcher based on the region (which knows function indices and have the right buffer handling wrappers).
* They all call the dispatcher with a (run-time) constant index, defined at compile time, and the memory buffer created on the fly.
* Serialisation and de-serialisation is on demand.
