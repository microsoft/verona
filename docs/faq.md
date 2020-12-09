---
layout: default
title: Frequently asked questions
---
# FAQ

## What is Project Verona?

Project Verona is a research project being run by Microsoft Research with 
academic collaborators at Imperial College London.
We are exploring research around language and runtime design for safe scalable
memory management and compartmentalisation.
The prototype here only covers the memory management aspects of the research.

## What research questions is Project Verona attempting to answer?

There are several areas we are investigating in Project Verona.
A few of the high-level questions are here:

* If we design a language without concurrent mutation,
  can we build scalable memory management?

* Can linear regions be used to remove the restrictions of per-object 
  linearity without sacrificing memory management?

* Can language level regions be used to support compartmentalisations?

We are at different points in our research for these questions, and will submit
academic articles for peer review in due course.

## How is Project Verona related to other languages?

Project Verona is a research language that is inspired by ideas from other languages:

* [Rust](https://www.rust-lang.org)
* [Cyclone](http://cyclone.thelanguage.org/)
* [Pony](https://www.ponylang.io/)

Many of the ideas we built on have been popularised by Rust, such as borrowing
and linearity, and Pony, such as reference capabilities.

## Why did you not start by extending an existing language?

Backwards compatibility adds a lot of impedance and constraints on research.
With Project Verona we have chosen to start with a clean slate to see what 
can be achieved towards our research questions.
The ideas we develop hopefully can be tamed
and fed back into existing languages in a backwards compatible way.

## When will this be a product?

This is a research project not a product.
Sometimes research projects turn 
into products, e.g. F#, but normally they do not! We hope the ideas we research
can benefit other languages.

## Does Project Verona mean Microsoft is no longer using C++/C#/Rust/...?

Project Verona is a research project that is not affecting engineering choices 
in the company.

The Project Verona team is connected to the people using all the major languages 
at the company, and want to learn from their experience, so we can research the 
problems that matter.

## Why have you open sourced the project so early?

We are open sourcing at this stage to collaborate with academic partners on
the language design.
We want to conduct this research openly to benefit
the community in general.


## Why have you implemented Project Verona in C++ rather than a safe language?

This is really two questions.

### Why is the Verona runtime implemented in C++?

The runtime is inherently using a lot of unsafe code: it is producing the abstraction from the raw bits and bytes into the abstraction that the language uses.
It is also inherently racy providing numerous lock-free datastructures for messaging and scheduling work.
The runtime is also providing memory management concepts:

* the allocator, [snmalloc](https://github.com/microsoft/snmalloc), we designed for the runtime
* the management of regions of memory
* reference counting of various runtime concepts
* ...

Hence, the implementation requires very low-level access to the machine, that cannot be found in any safe language that we know of.
When we started the project, C++ has the best tooling for handling unsafe code.
Rust would be an interesting choice to understand what abstraction we could surface to Rust.
As the concepts we are surfacing are different to Rust's type system it is unclear how beneficial this would be.

Ultimately, we want to verify the runtime against a formal specification, but this is a massive undertaking and is on the boundary of current verification research.


### Why is the Verona compiler implemented in C++?

One of our core aims with Project Verona is to support high-quality C++ FFI.
To support this, we will need extremely tight integration with a C++ compiler, like Clang.
Using C++ as the implementation language of our compiler makes this integration much
simpler.

Self-hosting the front-end of the compiler is a long-term possibility.
