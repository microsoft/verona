# Systems programming

The term system programming language is used to cover a wide range of problems from high-level performance-critical systems going down the stack to low-level memory managers and kernel modules.
There are two distinct aspects to system programming:

* Predictability
  - Latency
  - Resource usage

* Raw access 
  - Can treat memory directly as bits and bytes
  - Little or no abstraction on the hardware

To implement a low-level system of various kinds (for example, a memory manager), you need raw access.
In some sense, the memory manager is producing an abstraction on the machine that high-level services can consume.
Guaranteeing safety through a type system for programmers with raw access has not been achieved.

The "predictability" that higher-level systems require, however, we believe can be safely achieved with type systems.

With Project Verona, we are carving out an area of system programming, "infrastructure programming", that has important performance and predictability requirements, without needing raw access to the machine.

In Project Verona, we do not expect our research language to be suitable for implementing the Verona runtime itself.
This is an explicit non-goal of the project. 

# Concurrent Mutation and memory safety

There is a tension in programming language design between scalability and safety.
To provide temporal memory safety (no use after free bugs), you need to know when an object will never be accessed again.
This is, however, typically a global problem as the accesses can be across many threads and determining when an object is no longer accessed requires some form of consensus between the threads.
This is what a global GC does with its stop-the-world phases, which scan memory to determine if something is unreachable.
This dynamic consensus that an object is no longer in use can cause 
either latency issues or throughput.
There is amazing research in doing this better
but in Verona we want to take an alternative approach: ownership.

Ownership has been used in a few languages to support scalable memory management.
Rather than allowing multiple threads to access a single object, we ensure that at most one thread can access an object at a time.
This is fundamental in languages like [Pony](https://www.ponylang.io) and [Rust](https://www.rust-lang.org).
Ownership removes the need for dynamic consensus.
It provides static consensus, a single thread is responsible for deallocating an object.
Static ownership is transferable, but such a transfer is a two-party consensus problem rather than a global consensus problem.
Thus much easier to implement efficiently.

In Project Verona, we view giving up concurrent mutation as a necessary step for scalable memory management.
By eliminating concurrent mutation, we cannot implement concurrency as a library.
There are two alternatives here, expose "unsafe" to enable unsafe libraries for concurrency (e.g. Rust), or provide a concurrency model as part of the language (e.g. Pony).

The former means the language can rely on fewer invariants as it does not understand how the code inside the unsafe blocks is providing concurrency.
The latter means you need an amazing concurrency story as you can only have one.


# Concurrent Ownership

In Project Verona, we are introducing a new model of concurrent programming: concurrent owners, or cowns for short.
A cown, pronounced like "cone", encapsulates some set of resources (e.g. regions of memory) and ensures they are accessed by a single thread of execution at a time.

In Verona, we can wrap an object in a `cown`, and that will make it concurrent.
```
// x is some isolated object graph
var c = cown.create(x)
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
The `when` keyword is not blocking: it asynchronously schedules work, and returns immediately.
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

There are two other slightly more advanced examples of concurrency that are well documented:
  * [Dining Philosophers](../testsuite/demo/run-pass/dining_phil.verona), which illustrates a slightly more elaborate concurrent protocol.
  * [Parallel Fibonacci](../testsuite/demo/run-pass/fib.verona), which illustrates fork/join parallism using `when` and `promises`.


# Regions

Verona uses regions, groups of objects, as its fundamental concept of ownership.
Rather than specifying object ownership as a reference owns an object,
we generalise this a little to a reference can own a region, 
where a region is a group of objects.
Within a region, any object can refer to any other object in that region.
There is no restriction on the topology.
When the owning reference to a region goes away then the entire region is collected.

We use `iso`, for isolated, in a type to mean this is an owning reference to a region.
We use `mut`, for mutable, in a type to mean a mutable but non-owning reference. 
When `mut` is used in a field type, the reference points to an object in the same region as the object with the field. 
When `mut` is used on an argument type, the reference points to an object in an unknown region.
This is similar to a borrowed type in Rust.

When we allocate an object, we specify if it should be in its own region:
```
var x = new Node;
```
or in the same region as another object
```
var y = new Node in x
```

Regions can be nested, and form a forest, where the roots are either on the
stack or in cowns. 

We have a collection of simple examples to understand how the mechanism works.
We recommend looking at [region101](../testsuite/demo/run-pass/region101.verona) first to understand the basics,
and then the implementation of a [queue](../testsuite/demo/run-pass/library/queue.verona),
and its uses in [queue harness](../testsuite/demo/run-pass/queue_harness.verona)
and a [simple scheduler](../testsuite/demo/run-pass/scheduler.verona).



# Systematic testing

Inspired by [P and P#](https://github.com/p-org/), the Verona runtime deeply integrates systematic testing into its design.
The concurrency model of Verona means that all concurrent interactions must be understood by the runtime. 
This means that we can produce a replayable interleaving.
This is surfaced in numerous runtime unit tests that ensure good coverage of the runtime, and run on every CI build.

The primary implementation has been targeted at testing the runtime, but we have surfaced an alternative interpreter for the language to aid in testing.
This is built in the `veronac-sys` and `interpeter-sys`. These take additional parameters of 
```
  --run-seed N
  --run-seed_upper N
```
So 
```
  veronac-sys.exe --run testsuite/demo/run-pass/dining_phil.verona --run-seed 100 --run-seed_upper 200
```
will run 100 example interleavings of the program.  If you replace the line in the program
```
    d.count = 4;
```
with 
```
    d.count = 3;
```
Then the program is incorrect: 
it only waits for 3 philosophers to finish eating before it checks how many times the forks have been used.
In standard running, this is very hard to observe as the interleaving is unlikely to occur.
But in systematic testing, we can observe the failure:
```
...
Seed: 122
...
fork 1: 19
fork 2: 20
fork 3: 20
fork 4: 19
philosopher 4 eating, hunger=1
philosopher 4 eaten, new hunger=0
philosopher 4 leaving
philosopher leaving, door count 0
philosopher left, door count 18446744073709551615
Seed: 123
...
```
That is because for seed 122 some of the forks aren't used enough times before the Door exits.
We can now replay the example adding additional logging to understand and debug the problem.
Running a particular seed can be done as:
```
  veronac-sys.exe --run testsuite/demo/run-pass/dining_phil.verona --run-seed 122
```

Note: the seed is consistent across platforms, 
but not necessarily across versions of the runtime.
This seed was checked with dbec63a0fa4969cb5e186773f626e94c2494811f.  
The precise semantics of what will be preserved across versions is still to be decided.

This feature has been invaluable in debugging some of the concurrent algorithms in the runtime,
and we believe surfacing to the language will be a huge benefit for testing Verona programs.
