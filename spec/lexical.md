## Lexical Elements

### Comments

### Identifiers and Symbols

- `_` as "don't care".

### Primitive literals

- Boolean
- Various integers and floats

> [TODO:] How to talk about integer literals not having a type yet?  That is, 0 is zero, not a particular type of 0.  https://www.haskell.org/tutorial/numbers.html  

### Array literals

### String literals

> [TODO:] String literals are not for a specific text encoding, like 0 is the zero it needs to be (float, signed, unsigned, 1/2/4/8/16/32/64bit, etc.).

### Lambdas

### Object Literals

> [Open issues:]  code reuse, possible conflict with "real" `new`.
> Note that the parser current uses `new { ... }` as an object literal, which might conflict with object allocation.

### Expression Termination

- Blank lines
- New lines
- Indentation
- Trailing lambdas
