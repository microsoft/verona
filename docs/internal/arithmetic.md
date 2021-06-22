# Arithmetic

Numeric types in Verona are nothing but classes with methods that implement known arithmetic functions.

For example:
```
  let x : I8 & imm = 42;
  let y : I8 & imm = 24;

  // This is what a programmer would write
  let sum = x + y; // type inference to I8

  // This is what it would turn to
  let sum = x.+(y);
  // Or this
  let sum = I8::+(x, y);
```

Operations in Verona are nothing but function calls with numeric types.
If an operation refers to a function that is not implemented, this is a syntax error.

For example:
```
  let x : I8 & imm = 42;
  let y : I16 & imm = 24;

  // There is no such function I8::+(I8, I16);
  let sum = x + y; // ERROR

  // We can extend x, then add
  let z : I16 & imm = x.toI16();
  let sum = z + y; // type inference to I16

  // This is what it would turn to
  let sum = z.+(y);
  // Or this
  let sum = I16::+(z, y);
```

Verona does not have type casts, auto-promotion or guessing the types of arguments to functions.

The implementation of `x + y` to `x.+(w)` isn't really a conversion, but a syntactic feature of the language.
Prefix and infix operations are handled in the same way, so `a foo b` is the same as `a.foo(b)` and `ATy::foo(a, b)`.

## Type-boxing

To represent dynamic types, we'll use type boxing, which uses the fact that IEEE-754 64-bit floating point numbers have 53 bits left for NaN encodings (aka. NaN-boxing).
With around 48 bits used to represent pointers, and any integer up to 32-bits fitting that space, too, we can use the left-over bits to tag which type the 64-bit pattern represents.
Obviously, 64-bit integers cannot be represented in that manner, so it will need a longer representation (tag+payload) on different words.

The implementation details are still under discussion, which will determine what type of boxing we want and which bits we'll use for what.

It is worth noting that this will also be architecture-dependent.

For example:
 * On CHERI we probably can't steal the high bits, but we can use the tag bit to differentiate between a pointer and 128 bits of not-pointer.
   This means that we could have a `U128` and any number of pointer types in the same union type using the same amount of space as a pointer.
 * On 32-bit systems we probably won't want to do NaN boxing for anything that doesn't include a f64 value.

This is relatively important because it means that we need to make sure that the compiler abstracts this representation as much as possible.
