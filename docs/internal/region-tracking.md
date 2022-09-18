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

The region associated with each `mut` also affects the calling convention.
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

