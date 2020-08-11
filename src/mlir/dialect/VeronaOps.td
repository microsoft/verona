// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_OPS
#define VERONA_OPS

include "VeronaDialect.td"
include "TypecheckInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Verona_CopyOp : Verona_Op<"copy", [ Typecheckable ]> {
    let summary = "Copy a reference";
    let description = [{
      Make a copy of the given reference.
    }];

    let arguments = (ins Verona_Type : $input);
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{$input attr-dict `:` type($input) `->` type($output)}];
    let verifier = [{ return success(); }];
}

def Verona_ViewOp : Verona_Op<"view"> {
    let summary = "Get an unowned reference to the specified object.";
    let description = [{
      Get an unowned reference to the specified object. This operation can be
      used on both owned and unowned references. In either case, an unowned
      reference to that object is returned.

      Unlike `verona.copy`, this operation never consumes its operand.
    }];

    let arguments = (ins Verona_Type : $input);
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{$input attr-dict `:` type($input) `->` type($output)}];
    let verifier = [{ return success(); }];
}

def Verona_FieldReadOp : Verona_Op<"field_read"> {
    let summary = "Read a field from a reference";
    let description = [{
      Read the value of a field from a reference. The result value is always an
      unowned reference, regardless of whether the field was an owned reference
      or not.

      The following example reads the value of field `f` from the reference at
      `%x`.

      ```mlir
        %y = verona.field_read %x["f"] : !verona.C -> !verona.D
      ```
    }];

    let arguments = (ins Verona_Type : $origin, StrAttr : $field);
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{$origin `[` $field `]` attr-dict `:` type($origin) `->` type($output)}];
    let verifier = [{ return success(); }];
}

def Verona_FieldWriteOp : Verona_Op<"field_write"> {
    let summary = "Write to a field through a reference";
    let description = [{
      Write a new value to a field. The field's old value is returned.
    }];

    let arguments = (ins
      Verona_Type : $origin,
      Verona_Type : $value,
      StrAttr : $field
    );
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{
        $origin `[` $field `]` `,` $value attr-dict `:` type($origin) `->` type($value) `->` type($output)
    }];
    let verifier = [{ return success(); }];
}

def Verona_TidyOp : Verona_Op<"tidy"> {
    let summary = "Perform garbage-collection on a region";
    let description = [{
      Perform garbage-collection in the specified region. The actual semantics
      depend on the kind of region. For example, in an arena region, this could be
      a no-op.
    }];

    let arguments = (ins Verona_Type : $region);
    let assemblyFormat = [{$region attr-dict `:` type($region)}];
    let verifier = [{ return success(); }];
}

def Verona_DropOp : Verona_Op<"drop"> {
    let summary = "Drop a value";
    let description = [{
      Drop a value, deallocating the referenced region if the value was an owned
      reference.
    }];

    let arguments = (ins Verona_Type : $region);
    let assemblyFormat = [{$region attr-dict `:` type($region)}];
    let verifier = [{ return success(); }];
}

def Verona_AllocateRegionOp : Verona_Op<"new_region"> {
    let summary = "Allocate an object in a new region";
    let description = [{
      Allocate a new object of the given class in a new region.

      The operation needs to be provided the list of field names, and for each
      field, the SSA value to use to initialize it.

      The returned value is an owned reference to the allocated object, which is
      also the entrypoint to the region.

      The following example allocates an object of class C in a new region. The
      fields f and g of the object are initialized respectively with the values
      %x and %y.

      ```mlir
        %w = verona.new_region @C["f", "g"](%x, %y : !verona.A, !verona.B) : !verona.C
      ```

      TODO: the syntax is a little ugly, something that looks like a record
      eg. `verona.new_region @C[ "f" = %x : !verona.A, "g" = %y : !verona.B ]`,
      binding field names to SSA values might be nicer. Probably needs a custom
      parser though.
    }];

    let arguments = (ins
      SymbolRefAttr:$class_name,
      StrArrayAttr:$field_names,
      Variadic<Verona_Type>:$fields
    );
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{
      $class_name $field_names
      (`(` $fields^ `:` type($fields) `)`)?
      attr-dict
      `:` type($output)
    }];
}

def Verona_AllocateObjectOp : Verona_Op<"new_object"> {
    let summary = "Allocate an object in an existing region";
    let description = [{
      Allocate a new object of the given class in an existing region.

      The operation needs to be provided the list of field names, and for each
      field, the SSA value to use to initialize it.

      The region in which to allocate the object specified through an additional
      operand. The operand only needs to be a reference to an object in that
      region; it does not need to be a reference to its entrypoint.

      The returned value is an unowned reference to the allocated object.

      The following example allocates an object of class D in the region of %z.
      The fields f and g of the object are initialized respectively with the
      values %x and %y.

      ```mlir
        %w = verona.new_object @D["f", "g"](%x, %y : !verona.A, !verona.B) in (%z : !verona.C) : !verona.D
      ```
    }];

    let arguments = (ins
      SymbolRefAttr:$class_name,
      StrArrayAttr:$field_names,
      Variadic<Verona_Type>:$fields,
      Verona_Type:$region
    );
    let results = (outs Verona_Type : $output);

    let assemblyFormat = [{
      $class_name $field_names
      (`(` $fields^ `:` type($fields) `)`)?
      `in` `(` $region `:` type($region) `)`
      attr-dict
      `:` type($output)
    }];
}

def Verona_ClassOp : Verona_Op<"class", [
  Symbol, IsolatedFromAbove, SymbolTable,
  SingleBlockImplicitTerminator<"ClassEndOp">,
  NoRegionArguments
]> {
    let summary = "Verona class definition";
    let description = [{
      Operation used to define a Verona class. It contains a single region with a
      single block. The contents of that block describe the members of the class,
      ie. its fields and methods.

      This operation would typically be used directly at the module level.

      Example:

      ```mlir
        module {
          verona.class @C {
            verona.field "f" : !verona.U64;
          }
        }
      ```
    }];

    let arguments = (ins StrAttr : $sym_name);
    let results = (outs);
    let regions = (region SizedRegion<1> : $body);
}

// Trait used for operations that can only appear directly inside a class, such
// as field definitions.
def InClassScope : HasParent<"ClassOp">;

def Verona_ClassEndOp : Verona_Op<"_class_end", [Terminator, InClassScope]> {
    let summary = "End of a Verona class block";
    let description = [{
      This operation provides a dummy terminator used to end the block inside a
      `verona.class` operation. It has no purpose other than satisfying MLIR
      invariants about what a block looks like. It is implicit and does not need
      to be specified in textual representiations.
    }];

    let arguments = (ins);
    let results = (outs);
    let verifier = [{ return success(); }];

    let parser = ?;
    let printer = ?;
}

def Verona_FieldOp : Verona_Op<"field", [InClassScope]> {
    let summary = "Field definition";
    let description = [{
      This operation is used inside class bodies to define new fields.
    }];

    let arguments = (ins StrAttr : $name, Verona_TypeAttr : $type);
    let assemblyFormat = [{ $name `:` $type attr-dict }];
    let verifier = [{ return success(); }];
}

#endif // VERONA_OPS
