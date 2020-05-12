// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_OPS
#define VERONA_OPS

include "VeronaDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// This is a useless Verona op that accepts any Verona type as input and output.
// The purpose of it here is to test a dialect specific operation in the IR,
// ie. without quotes: verona.foo instead of the opaque "verona.foo".
def Verona_FooOp : Verona_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `verona.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = verona.foo %0 : i32
        ```
    }];

    let arguments = (ins Verona_Type:$input);
    let results = (outs Verona_Type:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // VERONA_OPS
