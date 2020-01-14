# Systems programming

The term system programming language is used to cover a wide range of problems from high-level performance critical systems going down the stack to low-level memory managers and kernel modules.
The are two distinct aspect to system programming:

* Predictability
  - Latency
  - Resource usage

* Raw access 
  - Can treat memory directly as bits and bytes
  - Little or no abstraction on the hardware

To implement many low-level systems, for instance a memory manager, you need raw access.
In some sense, the memory manager is producing an abstraction on the machine that high-level services can consume.
Guaranteeing safety through a type system for programmers with raw access has not been achieved.

The "predictability" that higher-level systems require, however, we believe can be safely achieved with type systems.

With Project Verona, we are carving out an area of system programming, "infrastructure programming", that has important performance and predictability requirements, without needing raw access to the machine.

In Project Verona, we do not expect our research language to be suitable for implementing the Verona runtime itself.
This is an explicit non-goal of the project. 

# Concurrent Mutation and memory safety

There is a tension in programming language design between scalability and safety.
To provide temporal memory safety (no use after free bugs), you need to know when an object will never be accessed again.
This is however, typically, a global problem as it can be across many threads and determining when it is no longer accessed requires some form of consensus between the threads.
This is what a global GC does with its stop the world phases, which scan memory to determine if something is unreachable.
This dynamic consensus that an object is no longer in use can cause 
either latency issues or throughput.
There is amazing research in doing this better
but in Verona we want to take an alternative approach: ownership.

Ownership has been used a in a few languages to support scalable memory management.
Rather than allowing multiple threads to access a single object, we ensure that at most one thread can access an object at a time.
This is fundamental in languages like [Pony](www.ponylang.io) and [Rust](https://www.rust-lang.org).
Ownership removes the need for dynamic consensus.   
It provides static consensus, a single thread is responsible for deallocating an object.
Static ownership is transferable, but such a transfer is a two-party consensus problem rather than a global consensus problem.
Thus much easier to implement efficiently.

We view Verona as giving up on concurrent mutation as a necessary step for scalable memory management.
By giving up on concurrent mutation, we cannot implement concurrency as a library.
There are two alternatives here, expose "unsafe" to enable unsafe libraries for concurrency (e.g. Rust), or provide a concurrency model as part of the language (e.g. Pony).

The former means the language can rely on fewer invariants as it does not understand how the code inside the unsafe blocks is providing concurrency.
The latter means you need an amazing concurrency story as you can only have one.


# Concurrent Ownership

In Project Verona, we are introducing a new model of concurrent programming: concurrent owners, or cowns for short.
A cown, pronounced like "cone", encapsulates some set of resources (e.g. regions of memory) and ensures they are accessed by a single thread of execution at a time.

In Verona, we can wrap an object in a `cown`, and that will make it concurrent.
```
// x is some isolated object graph
var c = cown(x)
// c is a cown that mediates access to x.
// We have lost direct access to x here
```

Once, we have wrapped an object in a cown, we can only access it by scheduling work on it.
In Verona, that is done with the keyword `when`
```
when (var x = c)
{
  // Access internals of cown(c) using name x in here
  Builtin.print("Hello\n")
}
Builtin.print("Goodbye\n")
```
The `when` keyword is not blocking it asynchronously schedules work, and returns immediately.
Thus it is perfectly possible for this to print
```
Goodbye
Hello
```
or
```
Hello
Goodbye
```
The typing of `when` effectively converts access of a `cown` to the type it wraps:
```
// c : cown[T]
when (var s = c)
{
  // s : T & mut
  ...
}
```
As a shorthand, we can omit the binding of the new variable and reuse the original 
variable inside the body:
```
// c : cown[T]
when (c)
{
  // c : T & mut
  ...
}
```

The test suite contains a simple example of using a `cown`, [bank1](../testsuite/demo/run-pass/bank1.verona).
This example adds an amount to a balance on a bank account. 

The concurrent owners add to the expressiveness of other approaches by enabling acquiring multiple cowns in one go.
Reading the demo examples [bank2](../testsuite/demo/run-pass/bank2.verona) and [bank3](../testsuite/demo/run-pass/bank3.verona) can help illustrate this.



# Regions

...
