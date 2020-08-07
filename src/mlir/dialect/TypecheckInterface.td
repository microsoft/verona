// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

include "mlir/IR/OpBase.td"

def TypecheckInterface : OpInterface<"TypecheckInterface"> {
  let description = [{
    Provides a hook for operations to define language-specific typechecking rules.

    This is, in spirit and in signature, similar to a custom MLIR verifier
    found in many operations. However we use a different hook for two principal
    reasons:
    - The verifier is expected to always succeed, at every stage of the
      compiler. In principle, in a bug-free compiler, the IR in memory should
      never fail the verifier. Typechecking on the other hand is not used until
      type inference has run. Indeed, until inference is run, it is not a bug
      for the IR to be ill-typed or incompletely-typed.

    - Typechecking could potentially be too expensive to run between every pass
      by default. Verifiers on the other hand are reasonably cheap, and could,
      in debug mode, run between every pass. In the future we'll have a way to
      explicitly opt-in to typechecking between every pass.

    Just like verifiers, the result of typechecking is not intended for user
    consumption. Typechecking exits on the first error and is not expected to
    print user-friendly error messages. The user-facing part of the type system
    will be provided by the type inference pass.

    For the time being, not all operations need to provide this interface. Those
    that don't are considered always valid. In the future, we will most likely 
    require all statement-like operations to implement this interface.
  }];
  let methods = [
    InterfaceMethod<
      "Check the types of this operation against typing rules.",
      "LogicalResult", "typecheck"
    >,
  ];
}

// Typecheckable can be used in the list of traits to opt-in to the
// TypecheckInterface, and have the `typecheck` method's signature automatically
// declared. Of course, an implementation of the method must still be provided
// manually.
def Typecheckable : DeclareOpInterfaceMethods<TypecheckInterface>;
