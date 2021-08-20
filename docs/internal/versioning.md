Versioning in Verona
====================

**WARNING** This is an under-discussion proposal, not a final design for the language.

Verona, as an infrastructure language, aims to support large-scale software engineering, where programs may be built from different components with different release schedules and that are maintained by different teams.
A Verona programmer should have the expectation that updating to a new version of a package will not break their code.
A Verona programmer publishing a library should have tooling that improves their ability to make this guarantee for their downstream consumers.

The key goals for versioning are:

 - A programmer should be able to ship breaking changes to an API with backwards-compatibility shims and graceful deprecation.
 - Updating to a new version of a dependency should not cause code to fail to compile unless your code depends on a version of the interface that has been explicitly dropped in the new version.
 - Developers should be given tooling at warns them of API breakages and allows them to provide compatibility interfaces.
 - If a library ships a new version of an interface, downstream consumers should be able to gradually upgrade their codebase in the common case (unless the new API provides fundamentally different abstractions).

Visibility is a software-engineering feature
--------------------------------------------

Many object-oriented languages provide some forms of restrictions on when the fields or methods of an object can be accessed.
The Java security model depends on these.
In Verona, visibility is regarded as purely a software-engineering tool for defining API-stability guarantees.
As such, Verona provides only two levels of visibility:

 - Public: may be accessed by anything
 - Package-private: may be accessed only by code written in the same package

These visibility rules apply to methods, fields, and types.
They exist to establish a boundary for API-breaking changes.
Adding or removing a private field or method does not constitute an API break.
Changing the behaviour or a private method does not constitute an API break.
Adding or removing a private type does not constitute an API break.
All of these are true because no code outside of the package may name the types, fields, or methods that have changed.

API-breaking changes
--------------------

Several kinds of change in Verona can introduce API-breaking changes.
Note that we differentiate between *API breakage* and *implementation breakage*.
Nothing that we do can prevent a buggy version of a package from being shipped.
Even within the category of interface breakage, there are two key subcategories:

 - Changes that affect things that are visible in the Verona type system and can cause a consumer of a package to fail to compile.
 - Changes to behaviour documented via API contracts that can cause a consumer of a package to behave unexpectedly.

For example, consider a map defines an iterator that returns the elements in insertion order.
If a later version of the same package redefines this API to return the elements sorted by a total ordering on the keys, this is a different type.
The API in both versions appears identical at the level of an abstract syntax tree but moving to the new version will break most consumers.
We aim to provide tools to help developers avoid this kind of problem but we cannot (and do not try to) detect it automatically.

There are a lot of ways of breaking a Verona API such that some consumers will fail to compile.
These are [*Note* This should be an exhaustive list, but probably isn't.  We should add everything else we discover here]:

 - Removing a public method or field from a class that is public.
   This breaks any code that tries to call the method / access the field or that tries to cast a pointer to that concrete type to an interface type defining that member.
 - Removing a public method field from a private class that is returned as an interface type from a public method.
   Code outside the package may cast this to an interface defining the member and expose it.
 - Adding a public method or field to a public class.
   This will potentially allow the class to match interfaces that it did not previously and may affect the behaviour of code that assumes specific semantics from classes that match the interface.
   It will also cause compile failures for any code that uses implementation inheritance to reuse the code from that class and defines methods or fields of the same name.
 - Adding any top-level public field, method, or type.
   This will break any file that imports the module into its top-level scope and defines something with the same name (or imports something else into its top-level scope that defines something with the same name).

In most packages, it will be important to make this kind of breaking change periodically.
The simplest way for a package author to do this is simply to bump the library version and require everyone to fix any breakages in their code.
This is not desirable because the only way that this allows *graceful deprecation*, which provides a window in which old and new versions of the APIs exist, is to ship two versions of the library.
Shipping two versions of the library causes several problems, most notably:

 - Security fixes must be back-ported to the old code.
   This is somewhat tractable when you have two versions but if this is needed for every API change then it can rapidly become dozens of versions.
 - If a project uses package A and also package B that uses package A, then it is possible to end up with a dependency on two incompatible versions of package A.

API Versioning
--------------

In Verona, we differentiate between the package (implementation) version and the API version.
A package version is a simple monotonic counter of package releases.
For each release, the package supports a set of API versions.
API versions use semantic versioning, where the major version is incremented to indicate the removal of types or methods and the minor version to indicate the addition of new classes or methods.
Both indicate potentially breaking changes for Verona consumers but it should be possible for a package to support all minor versions of an API with minimal effort by the package author.
There is also one special API version: `unstable`.
This represents any work-in-progress changes since the last release and exists so that developers can import the current snapshot of a library to test whether in-development changes are likely to cause problems.

By making the API version distinct from the implementation version, we allow:

 - A program to consume two different API versions of the same package (while consuming a single implementation)
 - A package to deprecate and remove APIs gradually over time without needing two versions of the code.
 - A package to add new APIs without breaking any consumers.

When a package is imported, the string describing the package should include the API version, for example:

```verona
import "some/package@2.0" as p;
```

The build tooling should pick the most recent version of the package that supports all of the versions that are used in the project and its dependencies and warn if that is not the most recent available version.
**Open question: Should pinning to specific implementations be part of this or should it be separate build metadata such as vcpkg.json?**

When the package is imported with this version, every class, interface, method, or field that is declared as public but is *not* exposed by this version is implicitly package-private.
This means that it cannot name any type or field.

Implementation
--------------

Package-private interface and class types are simply not nameable and therefore not usable in other packages.
The same is true if they are implicitly hidden by belonging to a version that is not in use.
This is purely a compile-time check.

Methods and fields are slightly more complex.
Methods or fields accessed on concrete types are trivially allowed or disallowed depending on whether the caller can see an API version that includes them.
Method or field accesses via interfaces depend on whether the selector used in the calling package unifies with the versioned selector in the callee's package.

Package-private members effectively introduce an anonymous (or, at least, unnameable) namespace for selectors such that `_foo` in one package and `_foo` in another are different selectors and there is no way of code in one package naming the other selector.
Versions simply generalise this slightly.
Versioned members are identified by a selector that includes the version and package of origin, which has the same value as a selector of the same name and arrity in a package that imports them and is permitted to see that version.


Corner cases / Open questions
-----------------------------

There are some corner cases that may have interesting implications here, in particular when objects are passed between modules.
For example, consider a package `A` that exposes the following class (straw-man syntax for declaring versions, probably not what we want to end up with):

```verona
class A
{
	// This is always public
	isSomething() : Bool;
	// This is public in any API version >= 2.0
	[[version=2.0+]]
	isSomethingElse() : Bool;
}
```

If a file in another package imports this then it may do something like:

```verona
import "A@1.0" as A;

interface IsSomethingElse
{
  isSomethingElse() : Bool;
};

var a = A.A();
```

The variable `a` here does not implement the interface `IsSomethingElse` because the `isSomethingElse` method is not visible here.
If `a` is passed to another package containing a similar interface, should it match?
There are two obvious answers:

 - If the other package imports `A` with a version <2.0, then it should not see the method.
 - If the other package imports `A` with a version >=2.0, then it should see the method.

The third case is the most interesting.
What is the correct behaviour if the other package does not import `A` at all?
Note that, without full dataflow analysis, it is not possible to tell in the general case which package the object was allocated in (for example, it might be stored in an array of `Any` allocated by a fourth module and populated by many other things in the system) and so the question reduces to selector identity.
Specifically: Is the selector `isSomethingElse/1` in a module that does not import `A` the same as the selector `isSomethingElse@2.0/1` in `A`?
There are at least two possible answers here:

 - If a module does not directly import another, it sees all versioned symbols as if they were public.
 - If a module does not directly import another, it does not see any versioned symbols.

The second option is almost certainly too limiting because it effectively prevents any reliance on structural typing across packages if versioning is used.
The first option potentially introduces breaks but is likely to avoid most of the common causes of breakage because the code in the module that is receiving objects via interface types and which cannot explicitly name the concrete types that they represent must be robust in the presence of new concrete types.
This is more obvious when you consider the transitive case:

```verona
// In Module A:
class A
{
  ...
  [[version=1.1+]]
  foo(Self & mut)
};

// In module B:
interface Fooable
{
  foo(Self & mut);
}

interface Any {}

getFooable(a : Any & mut) : Fooable | None
{
  match (a) {
    Fooable => a;
    _ => None
  }
}

// In module C
import "A@1.0" as A;
import "B@2.0" as B;

// Does this return `None` or a `Fooable`?
var a = B.getFooable(A.A());
```

If `a` at the end of this example is a `Fooable` then this is probably fine.
The surrounding code knows that `foo` is a method on an interface that can be returned from this function, even if it didn't get one with the previous version of package `A`.

The slightly more confusing outcome occurs if module `B` imports `A` with version 1.0, but `C` is updated to import version 1.1.
Now, `B` will not see the `foo` selector in the object as having the same identity as its `foo` selector and so the match will fail and `getFooable` will return `None`.

Tooling and workflow
--------------------

Versions require some tooling support.
A package's development is expected to follow a workflow that alternates between releases as snapshots of the codebase and under-development versions in between.
The tooling must not require that every single `git` commit provides API compatibility guarantees.
The expected workflow is:

 1. Developers work on a package and add / remove features while it is in development.
    During this process, the package metadata in the repository indicates that it is an unreleased version.
    This is not guaranteed to be API-compatible with *any* other version and may be imported only with the `unstable` version identifier.
 2. At some point, the code is stable and a release process begins.
 3. The tooling identifies any public symbols that were removed or added since the last release.
 4. If any symbols are removed, the tooling removes any API versions that included them from the list of supported versions.
 5. The tooling defines a new API version for the new release and updates all public symbols with version metadata.
 6. The release engineering team checks the output of the tool and:
   - Marks things that were not intended to be part of the public API as private.
   - Adds compatibility implementations for any accidentally removed symbols that caused support for API versions that were not intended to be broken to be dropped (marked explicitly with a sunset version identifier, so that they are not exposed in new versions of the API)
   - Adds any deprecation metadata required for APIs that will be removed in future versions.
   - Reruns the tooling until the release is in a happy state, supporting all of the expected API versions.
 7. The tool generates a complete list of the symbols in the new API version that can be committed to the repository and used with the next release.
 8. The package metadata is updated to indicate a supported release and tagged.

The tooling requirements follow from this workflow.
Tooling must be able to create a list of all symbols in a particular API version and check the code for conformance.
It must be able to add inline annotations with the version of the API where things were added.

Related work
------------

In a purely procedural API that takes only primitive types (including opaque pointers) as arguments, versioning is largely a solved problem.
C standard library implementations have used *symbol versioning* for many years.
With ELF symbol versioning, each C function name is transformed into a pair of a name and a version at static-link time.
During dynamic linking, multiple versions of the same symbol name can exist and the one whose version matches the one that the program expects will be selected.
This allows, for example, the `stat` call to support a different-sized structure depending on the version that the program expected.
This kind of API is much simpler than a Verona package interface because it does not expose any kind of structural types or any kind of subtyping relationship.


