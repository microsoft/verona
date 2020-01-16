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
* [Pony](https://github.com/ponylang/)

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
