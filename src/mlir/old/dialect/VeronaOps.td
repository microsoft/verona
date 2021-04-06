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

def Verona_FieldReadOp : Verona_Op<"field_read", [ Typecheckable ]> {
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

    let extraClassDeclaration = [{
      /// Get the type of the field being read by this operation, based of the
      /// type of the origin and the name of the field. 
      ///
      /// This method returns null if the field does not exist. This cannot
      /// happen on programs that typecheck successfully.
      Type getFieldType();
    }];
}

def Verona_FieldWriteOp : Verona_Op<"field_write", [ Typecheckable ]> {
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

    let extraClassDeclaration = [{
      /// Get the type of the field being read and written to by this operation,
      /// based of the type of the origin and the name of the field. Two
      /// different types are returned, as the read and the write types could
      /// differ (eg. when reading to an origin with a join type).
      ///
      /// This method returns a pair of nulls if the field does not exist. This
      /// cannot happen on programs that typecheck successfully.
      std::pair<Type, Type> getFieldType();
    }];
}

def Verona_CallOp : Verona_Op<"call"> {
    let summary = "Dynamic call to a class method";
    let description = [{
      The `$object` refers to the variable which represents the instance of a
      class, which has a method called `$name` with arguments (`self`, `$args`).

      Before type inference, the `$object` and most `$args` will probably have
      Verona.Uknown type, so this can't be a standard MLIR call.

      After type inference and name mangling for classes and methods, this could
      move to a simple call to the precise mangled name with the correct
      argument types.
    }];

    let arguments = (ins
      Verona_Type : $object,
      StrAttr : $name,
      Variadic<Verona_Type> : $args
    );
    let results = (outs Verona_Type : $res);

    let assemblyFormat = [{
        $name `[` $object `:` type($object) `]` `(` $args `:` type($args) `)` attr-dict `:` type($res)
    }];
    let verifier = [{ return success(); }];
}

def Verona_StaticOp : Verona_Op<"static"> {
    let summary = "Acquires a static descriptor of a type";
    let description = [{
      When applied upon a `$type` type, retrieves a descriptor for the class
      that will have enough information to know which static methods or fields
      it has.

      Returns a descriptor of the class to be used in field access and method
      calls.
    }];

    let arguments = (ins TypeAttr : $type);
    let results = (outs Verona_Type : $res);

    let assemblyFormat = [{
        $type attr-dict `:` type($res)
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

#endif // VERONA_OPS
