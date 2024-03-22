# Verona Language Spec

## Lexical Elements

### Comments

### Identifiers and Symbols

- `_` as "don't care".

### Strings

### Constants

- Boolean
- Various integers and floats

> How to talk about integer literals not having a type yet?

### Lambdas

### Object Literals

- Open issues: code reuse, possible conflict with "real" `new`.

### Expression Termination

- Blank lines
- New lines
- Indentation
- Trailing lambdas

## Program Structure

- It's all classes: modules are classes
- Fields
- Functions
- "Left-hand" functions
- Nesting
- Default values
- Visibility and encapsulation
- Generics: what needs to be discussed here as opposed to in `Types`?
  - Treat it as template-like, discuss type checking later
- Do we need to talk about type lists here?

## Expression Structure

### Bindings

- Parameters
- Let
- Var
- References

### Grouping with Parentheses

### Conditionals

### Static Accessors: Double Colons

### Dots

- Selectors
- Reverse application

### Universal Call Syntax

- Object-like vs. operator-like local names
  - Type names are `create` sugar
  - Function names are static calls
  - Unknown names are selectors
- Application: object-object adjacency
  - Apply sugar
- Prefix: operator-object adjacency
- Infix: object-operator-object adjacency
- Zero argument calls: lonely operators
- Tuple flattening
- Fields are property-like, treated as functions
- How to use this to write C-style, OO-style, and FP-style expressions

### Assignment

- Left and right side expressions
- Returning the previous value

### Partial Application

- Implicit.
- Explicit use of `_`.

### Non-local Returns

- In-lambda vs. not-in-lambda

### Try

- "Give me the value instead of returning"

### Pattern Matching

- Do we need to talk about type lists here?

## Types

### Classes as Types

- Closed world.

### Traits

- Open world.

### Type Aliases

### Tuples

- Anonymous classes.

### Algebraic Types

### Generics and Predicates

### Type Lists

- Generics for tuple arity.

### Capabilities

> Chicken and egg with regions.

### Function Types

- Sugar, and how.

## Type Inference and Checking

- Subtypes.

## Regions

- Spatial ownership.
- External uniqueness.
- Per-region memory management style.
- Resurrection-free finalization.
  - "Prompt drop".
  - Leads to Perceus-style for stack and RC regions.
  - GC regions are lazy.

## Concurrency

### Cowns

- "Shared regions".
- Temporal ownership.

### When

## Use

## Code Reuse

## Packages

### Core Library

- Types the compiler uses internally.

### Standard Library

- No "standard" library? Can we get away with great package management?

### Package Management

## FFI and ABI
