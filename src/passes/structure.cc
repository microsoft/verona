// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../lang.h"
#include "../wf.h"

namespace verona
{
  Node function_name(Node name)
  {
    if (!name)
      name = Ident ^ l_apply;
    else if (name == Symbol)
      name = Ident ^ name;

    return name;
  }

  PassDef structure()
  {
    return {
      "structure",
      wfPassStructure,
      dir::topdown,
      {
        // Field with a default value.
        // (equals (group let|var ident type) group)
        In(ClassBody) *
            (T(Equals)
             << ((T(Group)
                  << (T(Let, Var)[Let] * T(Ident)[Ident] * ~T(Type)[Type] *
                      End)) *
                 T(Group)++[Rhs])) >>
          [](Match& _) {
            Node node = _(Let) == Let ? FieldLet : FieldVar;
            return node << Explicit << _(Ident) << typevar(_, Type)
                        << (Block << (Expr << (Default << _[Rhs])));
          },

        // Field without a default value.
        // (group let|var ident type)
        In(ClassBody) *
            (T(Group)
             << (T(Let, Var)[Let] * T(Ident)[Ident] * ~T(Type)[Type] * End)) >>
          [](Match& _) {
            Node node = _(Let) == Let ? FieldLet : FieldVar;
            return node << Explicit << _(Ident) << typevar(_, Type) << DontCare;
          },

        // Function: `=` function after a `{}` function with no terminator.
        // (equals
        //  (group name typeparams params type llvmtype typepred brace ...)
        //  group)
        In(ClassBody) *
            (T(Equals)
             << ((T(Group)
                  << (~T(Ref)[Ref] * ~T(Ident, Symbol)[Ident] *
                      ~T(Square)[TypeParams] * T(Paren)[Params] *
                      ~T(Type)[Type] * ~T(LLVMFuncType)[LLVMFuncType] *
                      ~T(TypePred)[TypePred] * T(Brace)[Block] *
                      (Any * Any++)[Lhs])) *
                 T(Group)++[Rhs])) >>
          [](Match& _) {
            return Seq << (Function
                           << Explicit << (_(Ref) ? Lhs : Rhs)
                           << function_name(_(Ident))
                           << (TypeParams << *_[TypeParams])
                           << (Params << *_[Params]) << typevar(_, Type)
                           << (_(LLVMFuncType) || DontCare)
                           << typepred(_, TypePred) << (Block << *_[Block]))
                       << (Equals << (Group << _[Lhs]) << _[Rhs]);
          },

        // Function: f[T](x: T = e): T = e
        // (equals (group name typeparams params type llvmtype typepred) group)
        In(ClassBody) *
            (T(Equals)
             << ((T(Group)
                  << (~T(Ref)[Ref] * ~T(Ident, Symbol)[Ident] *
                      ~T(Square)[TypeParams] * T(Paren)[Params] *
                      ~T(Type)[Type] * ~T(LLVMFuncType)[LLVMFuncType] *
                      ~T(TypePred)[TypePred] * End)) *
                 T(Group)++[Rhs])) >>
          [](Match& _) {
            return Function
              << Explicit << (_(Ref) ? Lhs : Rhs) << function_name(_(Ident))
              << (TypeParams << *_[TypeParams]) << (Params << *_[Params])
              << typevar(_, Type) << (_(LLVMFuncType) || DontCare)
              << typepred(_, TypePred)
              << (Block << (Expr << (Default << _[Rhs])));
          },

        // Function: f[T](x: T = e): T { e }
        // (group name typeparams params type llvmtype typepred brace)
        In(ClassBody) * T(Group)
            << (~T(Ref)[Ref] * ~T(Ident, Symbol)[Ident] *
                ~T(Square)[TypeParams] * T(Paren)[Params] * ~T(Type)[Type] *
                ~T(LLVMFuncType)[LLVMFuncType] * ~T(TypePred)[TypePred] *
                ~T(Brace)[Block] * (Any++)[Rhs]) >>
          [](Match& _) {
            auto block = _(Block) ? (Block << *_[Block]) : DontCare;
            return Seq << (Function << Explicit << (_(Ref) ? Lhs : Rhs)
                                    << function_name(_(Ident))
                                    << (TypeParams << *_[TypeParams])
                                    << (Params << *_[Params])
                                    << typevar(_, Type)
                                    << (_(LLVMFuncType) || DontCare)
                                    << typepred(_, TypePred) << block)
                       << (Group << _[Rhs]);
          },

        // TypeParams.
        T(TypeParams) << (T(List)[TypeParams] * End) >>
          [](Match& _) { return TypeParams << *_[TypeParams]; },

        // TypeParam: (group ident)
        In(TypeParams) * T(Group) << (T(Ident)[Ident] * End) >>
          [](Match& _) { return TypeParam << _(Ident) << DontCare; },

        // TypeParam with default: (equals (group ident) group)
        In(TypeParams) * T(Equals)
            << ((T(Group) << (T(Ident)[Ident] * End)) * T(Group)++[Rhs]) >>
          [](Match& _) {
            return TypeParam << _(Ident) << (Type << (Default << _[Rhs]));
          },

        // ValueParam: (group ident type)
        In(TypeParams) * T(Group) << (T(Ident)[Ident] * T(Type)[Type] * End) >>
          [](Match& _) { return ValueParam << _(Ident) << _(Type) << Expr; },

        // ValueParam with default: (equals (group ident type) group)
        In(TypeParams) * T(Equals)
            << ((T(Group) << (T(Ident)[Ident] * T(Type)[Type] * End)) *
                T(Group)++[Rhs]) >>
          [](Match& _) {
            return ValueParam << _(Ident) << _(Type)
                              << (Block << (Expr << (Default << _[Rhs])));
          },

        In(TypeParams) * (!T(TypeParam, ValueParam))[TypeParam] >>
          [](Match& _) {
            return err(
              _(TypeParam), "Expected a type parameter or a value parameter");
          },

        T(ValueParam) >>
          [](Match& _) {
            return err(_(ValueParam), "Value parameters aren't supported yet");
          },

        // Params.
        T(Params) << T(List)[Params] >>
          [](Match& _) { return Params << *_[Params]; },

        // Param: (group ident type)
        In(Params) * T(Group)
            << (T(Ident, DontCare)[Ident] * ~T(Type)[Type] * End) >>
          [](Match& _) {
            auto id =
              (_(Ident) == DontCare) ? (Ident ^ _.fresh(l_param)) : _(Ident);
            return Param << id << typevar(_, Type) << DontCare;
          },

        // Param: (equals (group ident type) group)
        In(Params) * T(Equals)
            << ((T(Group)
                 << (T(Ident, DontCare)[Ident] * ~T(Type)[Type] * End)) *
                T(Group)++[Expr]) >>
          [](Match& _) {
            auto id =
              (_(Ident) == DontCare) ? (Ident ^ _.fresh(l_param)) : _(Ident);
            return Param << id << typevar(_, Type)
                         << (Block << (Expr << (Default << _[Expr])));
          },

        In(Params) * (!T(Param))[Param] >>
          [](Match& _) { return err(_(Param), "Expected a parameter"); },

        // Use.
        In(ClassBody, Block) * T(Group) << T(Use)[Use] * (Any++)[Type] >>
          [](Match& _) {
            return (Use ^ _(Use)) << (Type << (_[Type] || DontCare));
          },

        T(Use)[Use] << End >>
          [](Match& _) { return err(_(Use), "Can't put a `use` here"); },

        // TypeAlias: (equals (group typealias typeparams typepred) group)
        In(ClassBody, Block) * T(Equals)
            << ((T(Group)
                 << (T(TypeAlias) * T(Ident)[Ident] * ~T(Square)[TypeParams] *
                     ~T(TypePred)[TypePred] * End)) *
                T(Group)++[Rhs]) >>
          [](Match& _) {
            return TypeAlias << _(Ident) << (TypeParams << *_[TypeParams])
                             << typepred(_, TypePred)
                             << (Type << (Default << _[Rhs]));
          },

        In(ClassBody, Block) * T(TypeAlias)[TypeAlias] << End >>
          [](Match& _) {
            return err(_(TypeAlias), "Expected a `type` definition");
          },
        T(TypeAlias)[TypeAlias] << End >>
          [](Match& _) {
            return err(_(TypeAlias), "Can't put a `type` definition here");
          },

        // Class.
        // (group class ident typeparams type typepred brace ...)
        In(Top, ClassBody, Block) * T(Group)
            << (T(Class) * T(Ident)[Ident] * ~T(Square)[TypeParams] *
                ~T(Type)[Type] * ~T(TypePred)[TypePred] * T(Brace)[ClassBody] *
                (Any++)[Rhs]) >>
          [](Match& _) {
            return Seq << (Class << _(Ident) << (TypeParams << *_[TypeParams])
                                 << inherit(_, Type) << typepred(_, TypePred)
                                 << (ClassBody << *_[ClassBody]))
                       << (Group << _[Rhs]);
          },

        In(Top, ClassBody, Block) * T(Class)[Class] << End >>
          [](Match& _) {
            return err(_(Class), "Expected a `class` definition");
          },
        T(Class)[Class] << End >>
          [](Match& _) {
            return err(_(Class), "Can't put a `class` definition here");
          },

        // Default initializers. These were taken off the end of an Equals.
        // Depending on how many there are, either repack them in an equals or
        // insert them directly into the parent node.
        (T(Default) << End) >> ([](Match&) -> Node { return DontCare; }),
        (T(Default) << (T(Group)[Rhs] * End)) >>
          [](Match& _) { return Seq << *_[Rhs]; },
        (T(Default) << (T(Group)++[Rhs]) * End) >>
          [](Match& _) { return Equals << _[Rhs]; },

        // Type structure.
        TypeStruct * T(Group)[Type] >>
          [](Match& _) { return Type << *_[Type]; },
        TypeStruct * T(List, Paren)[TypeTuple] >>
          [](Match& _) { return Type << (TypeTuple << *_[TypeTuple]); },

        // Anonymous structural types.
        TypeStruct * T(Brace)[ClassBody] >>
          [](Match& _) {
            return Trait << (Ident ^ _.fresh(l_trait))
                         << (ClassBody << *_[ClassBody]);
          },

        // Strings in types are package descriptors.
        TypeStruct * T(String, Escaped)[Package] >>
          [](Match& _) { return Package << _(Package); },

        TypeStruct *
            (T(Equals,
               Arrow,
               Use,
               Class,
               TypeAlias,
               Var,
               Let,
               Ref,
               If,
               Else,
               New,
               Try,
               LLVMFuncType) /
             Literal)[Type] >>
          [](Match& _) { return err(_(Type), "Can't put this in a type"); },

        // A group can be in a Block, Expr, ExprSeq, Tuple, or Assign.
        In(Block, Expr, ExprSeq, Tuple, Assign) * T(Group)[Group] >>
          [](Match& _) { return Expr << *_[Group]; },

        // An equals can be in a Block, ExprSeq, Tuple, or Expr.
        In(Block, ExprSeq, Tuple) * T(Equals)[Equals] >>
          [](Match& _) { return Expr << (Assign << *_[Equals]); },
        In(Expr) * T(Equals)[Equals] >>
          [](Match& _) { return Assign << *_[Equals]; },

        // A list can be in a Block, ExprSeq, or Expr.
        In(Block, ExprSeq) * T(List)[List] >>
          [](Match& _) { return Expr << (Tuple << *_[List]); },
        In(Expr) * T(List)[List] >> [](Match& _) { return Tuple << *_[List]; },

        // Empty parens are Unit.
        In(Expr) * (T(Paren) << End) >> ([](Match&) -> Node { return Unit; }),

        // Empty expr is Unit.
        T(Expr) << End >> [](Match&) { return Expr << Unit; },

        // A tuple of arity 1 is a scalar.
        In(Expr) * (T(Tuple) << (T(Expr)[Expr] * End)) >>
          [](Match& _) { return _(Expr); },

        // A tuple of arity 0 is unit. This might happen through rewrites as
        // well
        // as directly from syntactically empty parens.
        In(Expr) * (T(Tuple) << End) >> ([](Match&) -> Node { return Unit; }),

        // Parens with one element are an Expr. Put the group, list, or equals
        // into the expr, where it will become an expr, tuple, or assign.
        In(Expr) * ((T(Paren) << (Any[Lhs] * End))) >>
          [](Match& _) { return _(Lhs); },

        // Parens with multiple elements are an ExprSeq.
        In(Expr) * T(Paren)[Paren] >>
          [](Match& _) { return ExprSeq << *_[Paren]; },

        // Typearg structure.
        (TypeStruct / In(Expr)) * T(Square)[TypeArgs] >>
          [](Match& _) { return TypeArgs << *_[TypeArgs]; },
        T(TypeArgs) << T(List)[TypeArgs] >>
          [](Match& _) { return TypeArgs << *_[TypeArgs]; },
        In(TypeArgs) * T(Group)[Type] >>
          [](Match& _) { return Type << *_[Type]; },
        In(TypeArgs) * T(Paren)[Type] >>
          [](Match& _) { return Type << *_[Type]; },

        // Object literal.
        In(Expr) * T(New) * T(Brace)[ClassBody] >>
          [](Match& _) {
            auto class_id = _.fresh(l_objlit);
            return Seq << (Lift << Block
                                << (Class << (Ident ^ class_id) << TypeParams
                                          << inherit() << typepred()
                                          << (ClassBody << *_[ClassBody])))
                       << (Expr << (Ident ^ class_id) << DoubleColon
                                << (Ident ^ l_create) << Unit);
          },

        // Lambda { [T](x: T = e, ...): T where T => ... }
        // (brace (group typeparams params type typepred) (group arrow) ...)
        In(Expr) * T(Brace)
            << ((T(Group)
                 << (~T(Square)[TypeParams] * T(Paren)[Params] *
                     ~T(Type)[Type] * ~T(TypePred)[TypePred])) *
                (T(Group) << T(Arrow)) * (Any++)[Rhs]) >>
          [](Match& _) {
            return Lambda << (TypeParams << *_[TypeParams])
                          << (Params << *_[Params]) << typevar(_, Type)
                          << typepred(_, TypePred) << (Block << _[Rhs]);
          },

        // Lambda: { a (, b...) => ... }
        // (brace (list|group) (group arrow) ...)
        In(Expr) *
            (T(Brace)
             << (T(List, Group)[Params] * (T(Group) << T(Arrow)) *
                 Any++[Rhs])) >>
          [](Match& _) {
            return Lambda << TypeParams << (Params << _[Params]) << typevar(_)
                          << typepred() << (Block << _[Rhs]);
          },

        // Zero argument lambda: { ... } (brace ...)
        In(Expr) * T(Brace) << (!(T(Group) << T(Arrow)))++[Rhs] >>
          [](Match& _) {
            return Lambda << TypeParams << Params << typevar(_) << typepred()
                          << (Block << _[Rhs]);
          },

        // Var.
        In(Expr) * T(Var)[Var] * T(Ident)[Ident] >>
          [](Match& _) { return Var << _(Ident); },

        T(Var)[Var] << End >>
          [](Match& _) { return err(_(Var), "`var` needs an identifier"); },

        // Let.
        In(Expr) * T(Let)[Let] * T(Ident)[Ident] >>
          [](Match& _) { return Let << _(Ident); },

        T(Let)[Let] << End >>
          [](Match& _) { return err(_(Let), "`let` needs an identifier"); },

        // Move a ref to the last expr of a sequence.
        In(Expr) * T(Ref) * T(Expr)[Expr] >>
          [](Match& _) { return Expr << Ref << *_[Expr]; },
        In(Expr) * T(Ref) * T(Expr)[Lhs] * T(Expr)[Rhs] >>
          [](Match& _) { return Seq << _[Lhs] << Ref << _[Rhs]; },
        In(Expr) * T(Ref) * T(Expr)[Expr] * End >>
          [](Match& _) { return Expr << Ref << *_[Expr]; },

        // Lift Use, Class, TypeAlias to Block.
        In(Expr) * T(Use, Class, TypeAlias)[Lift] >>
          [](Match& _) { return Lift << Block << _[Lift]; },

        // A Type at the end of an Expr is a TypeAssert. A tuple is never
        // directly
        // wrapped in a TypeAssert, but an Expr containing a Tuple can be.
        T(Expr) << (((!T(Type))++)[Expr] * T(Type)[Type] * End) >>
          [](Match& _) {
            return Expr << (TypeAssert << (Expr << _[Expr]) << _(Type));
          },

        In(Expr) * (TypeCaps / T(TypePred, Arrow, LLVMFuncType))[Expr] >>
          [](Match& _) {
            return err(_(Expr), "Can't put this in an expression");
          },

        // A Block that doesn't end with an Expr gets an implicit Unit.
        In(Block) * (!T(Expr))[Lhs] * End >>
          [](Match& _) { return Seq << _(Lhs) << (Expr << Unit); },

        // An empty Block gets an implicit Unit.
        T(Block) << End >> [](Match&) { return Block << (Expr << Unit); },

        // Remove empty and malformed groups.
        T(Group) << End >> ([](Match&) -> Node { return {}; }),
        T(Group)[Group] >>
          [](Match& _) { return err(_(Group), "Syntax error"); },
      }};
  }
}
