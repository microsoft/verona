// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_TYPES
#define VERONA_TYPES

include "VeronaDialect.td"

def JoinType : Verona_TypeDef<"Join"> {
  let description = [{
    Join types are unions between types (A | B).
  }];

  let parameters = (ins ArrayRefParameter<"Type", "Elements">:$elements);
  let genVerifyInvariantsDecl = 1;
}

def MeetType : Verona_TypeDef<"Meet"> {
  let description = [{
    Meet types are intersections between types (A & B).
  }];

  let parameters = (ins ArrayRefParameter<"Type", "Elements">:$elements);
  let genVerifyInvariantsDecl = 1;
}

def UnknownType : Verona_TypeDef<"Unknown"> {
  let description = [{
    Unknown types are derived types from operations that cannot define the type
    at lowering stage, but will later be replaced by other types during type
    inference.
  }];
}

def CapabilityType : Verona_TypeDef<"Capability"> {
  let description = [{
    Capability types represents properties of individual references eg.
    a `String & iso` and a `String & mut` could point to the same object.
   
    Isolated: An entry point to a new region. There can be more than one
    reference to the entrypoint, but only one of them can be Isolated ie.
    the others must be Mutable.
   
    Immutable: A stronger property than Read-Only. It guarantees that no
    mutable aliases to that object exist anywhere else.
  }];

  let parameters = (ins "Capability":$capability);
}

def DescriptorType : Verona_TypeDef<"Descriptor"> {
  let description = [{
    A static class descriptor type, used for access to static members of the
    class, including fields and methods.
  }];

  let parameters = (ins "Type":$describedType);
  let genVerifyInvariantsDecl = 1;
}

def ViewpointType : Verona_TypeDef<"Viewpoint"> {
  let description = [{
    Viewpoint is a view of a type through another type.
    
    For examples, reading a mut field from a imm object gives you a
    `viewpoint<mut, imm> = imm` reference.
  }];

  let parameters = (ins "Type":$leftType, "Type":$rightType);
  let genVerifyInvariantsDecl = 1;
}

#endif
