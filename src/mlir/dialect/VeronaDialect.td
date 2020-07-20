// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_DIALECT
#define VERONA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Verona dialect definition.
//===----------------------------------------------------------------------===//

def Verona_Dialect : Dialect {
    let name = "verona";
    let summary = "A prototype Verona language MLIR dialect.";
    let cppNamespace = "verona";
}

//===----------------------------------------------------------------------===//
// Base verona operation/type definitions.
//===----------------------------------------------------------------------===//

class Verona_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Verona_Dialect, mnemonic, traits> {
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
}

def Verona_IntegerType : DialectType<Verona_Dialect,
    CPred<"$_self.isa<::mlir::verona::IntegerType>()">, "integer type">,
    BuildableType<"$_builder.getType<::mlir::verona::IntegerType>()">;

def Verona_Type : AnyTypeOf<[Verona_IntegerType]>;

#endif // VERONA_DIALECT
