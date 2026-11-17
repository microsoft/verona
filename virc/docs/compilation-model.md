# Compilation Model

**Current:** `virc::Compilation` in `virc/model/compilation.h` is the shared,
output-neutral result of VIRC analysis. It retains the VIR root and source
paths, indexed definitions, function analysis state, type records, and
typechecker environments.

`FuncState` maps symbolic labels and locals to dense indices and stores
liveness state. `TypeInfo` represents compound arrays, cowns, references,
unions, and tuples using other type IDs. Name tables intern execution and debug
strings separately.

Optimization retains only functions reachable from `@main`, method tables, and
memo initializers. Function IDs are compacted afterward, with `@main` remaining
at ID zero.

The model does not contain VBC wire values or VBCI live values. VBC encoding
of this state is described under the [VBC emitter](../vbc/README.md); future
emitters own representation choices at their output boundaries.