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

def Verona_Type : DialectType<Verona_Dialect,
    CPred<"isaVeronaType($_self)">, "Verona type">;

def Verona_TypeAttr :
    Attr<And<
      [CPred<"$_self.isa<TypeAttr>()">,
       CPred<"isaVeronaType($_self.cast<TypeAttr>().getValue())">]>>
{
  let storageType = [{ TypeAttr }];
  let returnType = [{ Type }];
  let valueType = NoneType;
  let convertFromStorage = "$_self.getValue().cast<Type>()";
}

#endif // VERONA_DIALECT
