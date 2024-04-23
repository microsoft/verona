# Verona Language Spec

This document is a work in progress. It is intended to be a comprehensive guide to the Verona programming language. It is not intended to be a tutorial, but rather a reference for those who are already familiar with the language.  In time, we will provide other documents to motivate and "sell" the language design.

> [Notes from discussion:] Should contain "rationale subsections" to explain why things are the way they are.

## [Lexical Elements](lexical.md)

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

### Object allocation/construction

> [Notes from discussion:] 
> * Static initialisers in other languages, and when things happen.
> * Immutable static data initialisation, probably not a problem here.
> * Initialisation order is complex in some languages.
> * `create` as a factory method, and allocates after it has found all the bits. 
>   So create can fail, make sure we allow failure returns from create.

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

> [Notes from discussion:] Should this mention `Store` types?

### Partial Application

- Implicit.
- Explicit use of `_`.

### Non-local Returns

- In-lambda vs. not-in-lambda

### Try

- "Give me the value instead of returning"

### Pattern Matching

- Do we need to talk about type lists here?

## Region Model

> [Notes from discussion:] 
> * Describe the region model before the types to capture and enforce it.
> * Object model section before types - pretty pictures
> * Types turn pretty pictures into maths.

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
