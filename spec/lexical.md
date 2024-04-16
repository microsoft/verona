## Lexical Elements

This section explains the lexical elements of the Verona language.
It forms part of the [language specification](spec.md),
which is intended to be a comprehensive guide to the Verona programming language.

### Comments

Verona supports two types of comments.
The first type is a single line comment:
```
// Single line comments
```
These start with a `//` and continue to the end of the line.
The termination of the comment is the end of the line or the end of file, which ever occurs sooner.

The second type of comment is a block comment:
```
/* Block comments */
```
These comments begin with a `/*` and end with a `*/`.
They can span multiple lines.
```
/*
   This is a multi-
   line block comment.
 */
```

The block comment is nestable, meaning that you can have a block comment within a block comment.
```
/* Block comments can contain commented code.
   /* This is a block comment */
   f(x,y,z)
   /* with another block comment */
   g(x,y,z)
 */
```
There is no limit to the nesting depth of block comments.

Block comments must be correctly terminated.
That is every `/*` must have a corresponding `*/`.
A non-terminated block comment is a syntax error.

> [TODO:] Explain interaction with string literals.
>         Possibly, here or in string section.

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
