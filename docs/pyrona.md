---
layout: default
title: Pyrona - Fearless Concurrency for Python
---

# Pyrona - Fearless Concurrency for Python

As part of Project Verona, we have been developing a new ownership model for Python, called Lungfish.
This model is designed to provide a safe and efficient way to manage memory and concurrency in Python programs.
Pyrona is our experiments to develop this approach.


## Plan for Python

### Quick prototyping in FrankenScript

Modifying a production language is a complex task.
As an initial step, we have been developing a toy language called [FrankenScript](https://github.com/fxpl/frankenscript) that allows us to quickly prototype our ideas for region-based ownership.
FrankenScript is a small language that is designed to be easy to modify and extend.
It is based on the ideas of ownership and concurrency that we are exploring in Project Verona, but where all the checks are dynamic.

This prototype has given us confidence in the conceptual ideas behind our ownership model, and has allowed us to explore the design space of ownership in a dynamic language.

### Engaging with the Python community

Over the last two years, we have been engaging with the Faster CPython team at Microsoft as a sounding board for our ideas.
In May 2025, we will be taking our first steps into the boarder Python community at the [Python Language Summit](https://us.pycon.org/2025/events/language-summit/).
We will be presenting our ideas and seeking feedback from the core developers of the language.


### Steps to a new ownership model

Building an ownership model for Python is a complex task, and we are taking a step-by-step approach to ensure that we get it right.

The first step is actually to build a concept of deep immutability into Python.  This can be split into three parts:

* Deep Immutability: We are starting with a deep immutability model, we have been drafting a [PEP](https://github.com/TobiasWrigstad/peps/pull/8) to describe this model.

* SCC work and atomic RC

* Integration with message passing between sub-interpreters

Deep Immutability is a key part of any ownership model as we need concurrent threads to be able to share type information.
As types in Python are objects, we need a way to share these objects between threads.

Once the immutability model is in place, then we can start applying the region-based ownership model to Python.
This involves applying the ideas developed in the FrankenScript prototype to Python.
We have made progress on this in our prototyping ideas on CPython, but nothing is ready to try yet.

## FAQ

### What is Pyrona?

It is our approach to bringing region based ownership to Python.

### Why Python?

Python is the most popular programming language in the world.
With [PEP703](https://peps.python.org/pep-0703/), aka NoGil or "Free-Threaded" Python, Python is moving to a fully concurrent model.
This means a vast quantity of programmers will be potentially exposed to concurrency issues.
Bringing an ownership model to Python will help to make it easier to write concurrent programs, and to avoid the pitfalls of concurrency.

This is a perfect opportunity to influence the future of Python, and to help make programming experience better for everyone.

### Why not simply use Rust's ownership model?

Rusts ownership model is designed for a statically typed language, and restricts the type of object graphs that can be used in the language.
There is so much existing code in Python that the restrictions of Rust's ownership model would be too limiting.
We are designing a new ownership model based on regions that is designed to work with Python's dynamic typing and existing object graphs.

Our approach draws heavily from the experience of languages with ownership models like Rust, Cyclone and Pony, but is based completely on dynamic checks.  This completely alters the kind of things that can be checked in comparison to a statically typed approach.

### What has Project Verona learnt from Pyrona?

Performing ownership research in a dynamically typed language has been a massive learning experience for the Project Verona team.
It has challenged every assumption we have made about ownership and concurrency in a statically typed language.
Moreover, it has shown us certain programming patterns that are really easy to check in a dynamic region system, but are really hard to check in a statically typed language.
We are currently re-evaluating the balance between the static and dynamic checks in our ownership model, and how to best combine them.  Hopefully, we will be able to bring some of these ideas back into the Project Verona research language.

### Where can I find out more?

We have a detailed list of publications related to Project Verona on our [publications page](/publications.html).
The most relevant paper to Pyrona is [Dynamic Region Ownership for Concurrency Safety](https://www.microsoft.com/en-us/research/publication/dynamic-region-ownership-for-concurrency-safety/).

You can try our toy language, [FrankenScript](https://github.com/fxpl/frankenscript).

We are also working on several forks of Python that implement our ideas, these are currently on [GitHub](https://github.com/mjp41/cpython).

