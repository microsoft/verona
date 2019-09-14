# Verona type inference and checking

The implementation is split in a few modules:
- infer.{cc,h} contains the constraint generation from the SSA.
- solver.{cc,h} contains the solver which will attempt to find a substitution
  which satisfies all the constraints.
- constraints.{cc,h} describes the constraint reduction used by the solver,
  based on the subtyping rules of our language.
- typecheck.{cc,h} glues everything together and is the main entrypoint for
  typechecking.

The core algorithm is inspired by MLsub [0, 1], with a couple of extensions.

We remove the syntactical distinction between polar and negative types,
allowing unions to appear in inputs and intersections to appear in outputs.
This also allows invariant type parameters, in a way that is less instrusive
than as suggested in [1, Section 9.1.1].

The main issue this raises is how to solve constraints of the form
`(T1 | T2) <: T3` or `T1 <: (T2 & T3)`, as these can be solved in two different
ways, or even three if T3 (in the first case) or T1 (in the second case) are
inference variables. We circumvent this by keeping a backtracking stack of
all potential states during solving. The algorithm is very naive but
functional. In the future we may want to replacing the solver part by Z3.


While we've removed the distinction between polar and negative types, we
continue to distinguish between positive and negative occurences of
inference variables. To allow inference variables to be used in invariant
positions, we've introduced a "range" type syntax of the form `T1...T2`, where
`T1 <: T2`. When a fresh type variable is needed in an invariant position, we
use (-x, +x), where x is fresh.

The lower bound of a range can only contain negative variables and the upper
bound can only contain positive ones. A range which appears in a negative
position (such as in a subtyping constraint) can be replaced by its lower
bound and vice versa (as seen in the subtyping rules in constraint.h).

When inference is complete, we could replace ranges by either their lower or
upper bound, or even anything in between. We always take the lower bound,
as it is the most concrete type, which should have the most concise runtime
representation. This is usually also the most "intuitive" type.


We do not support recursive types and polymorphic lets bindings (other than
the top-level), even though MLsub does. There isn't any reason why we
couldn't support them, we just haven't needed them so far.

The results of the inference pass are written back into the AST.

[0]: "Polymorphism, Subtyping, and Type Inference in MLsub"
      https://www.cl.cam.ac.uk/~sd601/papers/mlsub-preprint.pdf
[1]: "Algebraic Subtyping"
      https://www.cl.cam.ac.uk/~sd601/thesis.pdf

