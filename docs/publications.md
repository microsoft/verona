---
layout: default
title: Publications for Project Verona
---
# Publications

This page contains the papers related to Project Verona. They are grouped by topic.

## Ownership

- [Dynamic Region Ownership for Concurrency Safety](https://www.microsoft.com/en-us/research/publication/dynamic-region-ownership-for-concurrency-safety/)
  Fridtjof Peer Stoldt, Brandt Bucher, Sylvan Clebsch, Matthew Johnson, Matthew J. Parkinson, Guido Van Rossum, Eric Snow and Tobias Wrigstad, PLDI 2025.
- [Reference Capabilities for Flexible Memory Management](https://www.microsoft.com/en-us/research/publication/reference-capabilities-for-flexible-memory-management/)
  Ellen Arvidsson, Elias Castegren, Sylvan Clebsch, Sophia Drossopoulou, Matthew J. Parkinson, James Noble and Tobias Wrigstad, OOPSLA 2024.

## Concurrency

The concurrency model develop by Project Verona is called Behaviour-Oriented Concurrency. It is described in the following paper:
- [When Concurrency Matters: Behaviour-Oriented Concurrency](https://www.microsoft.com/en-us/research/publication/when-concurrency-matters-behaviour-oriented-concurrency)
  Luke Cheeseman, Matthew J. Parkinson, Sylvan Clebsch, Marios Kogias, Sophia Drossopoulou, David Chisnall, Tobias Wrigstad and Paul Liétar, OOPSLA 2024.

It has been applied to build deterministic parallel execution in the following paper:
- [DORADD: Deterministic Parallel Execution in the Era of Microsecond-Scale Computing](https://doi.org/10.1145/3710848.3710872)
  Zhengqing Liu, Musa Unal, Matthew J. Parkinson and Marios Kogias, PPoPP 2025


## Reference counting
- [Reference Counting Deeply Immutable Data Structures with Cycles: an Intellectual Abstract](https://www.microsoft.com/en-us/research/publication/reference-counting-deeply-immutable-data-structures-with-cycles-an-intellectual-abstract/)
  Matthew J. Parkinson, Sylvan Clebsch and Tobias Wrigstad, ISMM 2024.
- [Wait-Free Weak Reference Counting](https://www.microsoft.com/en-us/research/publication/wait-free-weak-reference-counting/)
  Matthew J. Parkinson, Sylvan Clebsch and Ben Simner, ISMM 2023.

# snmalloc
snmalloc is a memory allocator that is used in Project Verona. You can read more about it in the following papers:

- [BatchIt: Optimizing Message-Passing Allocators for Producer-Consumer Workloads: An Intellectual Abstract](https://www.microsoft.com/en-us/research/publication/batchit-optimizing-message-passing-allocators-for-producer-consumer-workloads-an-intellectual-abstract/)
  Nathaniel Filardo, Matthew J. Parkinson, ISMM 2024.
- [snmalloc: A message passing Allocator](https://www.microsoft.com/en-us/research/publication/issm-2019-proceedings-of-the-2019-acm-sigplan-international-symposium-on-memory-management/)
  David Chisnall, Sylvan Clebsch, Sophia Drossopoulou, Juliana Vicente Franco, Paul Lietar, Matthew J. Parkinson, Alex Shamis and Christoph  M. Wintersteiger, ISMM 2019.

It is available on [Github](https://github.com/microsoft/snmalloc) and is used in many other projects, and has an active usage in the Rust community [snmalloc-rs](https://crates.io/crates/snmalloc-rs).


## Precursor works
Project Verona pulls together many strands of research from the past. The following papers are some of the most relevant to the work we are doing in Verona.

### Experiements in Manual Memory Management for .NET
-   [Project Snowflake: Non-blocking safe manual memory management in .NET](https://www.microsoft.com/en-us/research/publication/project-snowflake-non-blocking-safe-manual-memory-management-net/)
    Matthew J. Parkinson, Kapil Vaswani, Dimitrios Vytiniotis, Manuel Costa, Pantazis Deligiannis, Aaron Blankstein, Dylan McDermott and Jonathan Balkind, OOPSLA 2017.
-  [Simple, Fast and Safe Manual Memory Management](https://www.microsoft.com/en-us/research/publication/simple-fast-safe-manual-memory-management/)
    Piyus Kedia, Manuel Costa, Dimitrios Vytiniotis, Matthew J. Parkinson, Kapil Vaswani and Aaron Blankstein, PLDI 2017.

### The Pony Programming Language

- [Pony: Co-designing a Type System and a Runtime](https://www.ponylang.io/media/papers/codesigning.pdf)
  Sylvan Clebsch, Ph.D. Thesis, Imperial College London, 2017.
- [Deny capabilities for safe, fast actors](https://www.ponylang.io/media/papers/fast-cheap.pdf)
Sylvan Clebsch, Sophia Drossopoulou, Sebastian Blessing and Andy McNeil,  AGERE 2015.

There are many more papers on the Pony [website](https://www.ponylang.io/learn/papers/).

### Earlier papers

- [Uniqueness and Reference Immutability for Safe Parallelism](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2012-79.pdf)
  Colin S. Gordon, Matthew J. Parkinson, Jared Parsons, Aleks Bromfield and Joe Duffy, OOPSLA 2012.
  
- [External Uniqueness Is Unique Enough](https://doi.org/10.1007/978-3-540-45070-2_9)
  Tobias Wrigstad and Dave Clarke, ECOOP 2003.
