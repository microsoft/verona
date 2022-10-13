// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

#include "wf.h"

namespace sample
{
  auto err(NodeRange& r, const std::string& msg)
  {
    return Error << (ErrorMsg ^ msg) << (ErrorAst << r);
  }

  bool lookup(const NodeRange& n, std::initializer_list<Token> t)
  {
    auto def = (*n.first)->lookup_first();
    return def && def->type().in(t);
  }

  PassDef modules()
  {
    return {
      // Module.
      T(Directory)[Directory] << (T(File)++)[File] >>
        [](Match& _) {
          auto id = _(Directory)->location();
          return Group << (Class ^ _(Directory)) << (Ident ^ id)
                       << (Brace << *_[File]);
        },

      // File on its own (no module).
      In(Top) * T(File)[File] >>
        [](Match& _) {
          auto id = _(File)->location();
          return Group << (Class ^ _(File)) << (Ident ^ id)
                       << (Brace << *_[File]);
        },

      // Packages.
      T(Package) * (T(String) / T(Escaped))[String] >>
        [](Match& _) { return Package << _[String]; },

      T(Package)[Package] << End >>
        [](Match& _) {
          return err(_[Package], "`package` must have a descriptor string");
        },

      // Type assertion. Treat an empty assertion as DontCare. The type is
      // finished at the end of the group, or at a brace. Put a typetrait in
      // parentheses to include it in a type assertion.
      T(Colon) * ((!T(Brace))++)[Type] >>
        [](Match& _) { return Type << (_[Type] | DontCare); },
    };
  }

  inline const auto TypeStruct = In(Type) / In(TypeList) / In(TypeTuple) /
    In(TypeView) / In(TypeFunc) / In(TypeThrow) / In(TypeUnion) / In(TypeIsect);
  inline const auto Name = T(Ident) / T(Symbol);
  inline const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) /
    T(Hex) / T(Bin) / T(Int) / T(Float) / T(HexFloat);

  auto typevar(auto& _, const Token& t = Invalid)
  {
    auto n = _(t);
    return n ? n : Type << (TypeVar ^ _.fresh());
  }

  PassDef structure()
  {
    return {
      // Let Field:
      // (equals (group let ident type) group)
      // (group let ident type)
      In(ClassBody) *
          (T(Equals)
           << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
               T(Group)[rhs] * End)) >>
        [](Match& _) {
          return _(id = FieldLet)
            << _(id) << typevar(_, Type) << (Expr << *_[rhs]);
        },

      // (group let ident type)
      In(ClassBody) *
          (T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) >>
        [](Match& _) {
          return _(id = FieldLet) << _(id) << typevar(_, Type) << DontCare;
        },

      // Var Field:
      // (equals (group var ident type) group)
      // (group var ident type)
      In(ClassBody) *
          (T(Equals)
           << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
               T(Group)[rhs] * End)) >>
        [](Match& _) {
          return _(id = FieldVar)
            << _(id) << typevar(_, Type) << (Expr << *_[rhs]);
        },

      // (group var ident type)
      In(ClassBody) *
          (T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) >>
        [](Match& _) {
          return _(id = FieldVar) << _(id) << typevar(_, Type) << DontCare;
        },

      // Function: (equals (group name square parens type) group)
      In(ClassBody) *
          (T(Equals)
           << (T(Group) << (~Name[id] * ~T(Square)[TypeParams] *
                            T(Paren)[Params] * ~T(Type)[Type]) *
                 T(Group)[rhs] * End)) >>
        [](Match& _) {
          // auto name = _(id) ? _(id) : Ident ^ apply;
          _.def(id, Ident ^ apply);
          return _(id = Function)
            << _(id) << (TypeParams << *_[TypeParams]) << (Params << *_[Params])
            << typevar(_, Type) << (FuncBody << _[rhs]);
        },

      // Function: (group name square parens type brace)
      In(ClassBody) * T(Group)
          << (~Name[id] * ~T(Square)[TypeParams] * T(Paren)[Params] *
              ~T(Type)[Type] * ~T(Brace)[FuncBody] * (Any++)[rhs]) >>
        [](Match& _) {
          _.def(id, Ident ^ apply);
          return Seq << (_(id = Function)
                         << _(id) << (TypeParams << *_[TypeParams])
                         << (Params << *_[Params]) << typevar(_, Type)
                         << (FuncBody << *_[FuncBody]))
                     << (Group << _[rhs]);
        },

      // TypeParams.
      T(TypeParams) << T(List)[TypeParams] >>
        [](Match& _) { return TypeParams << *_[TypeParams]; },

      // TypeParam: (group ident type)
      In(TypeParams) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = TypeParam) << _(id) << typevar(_, Type) << Type;
        },

      // TypeParam: (equals (group ident type) group)
      In(TypeParams) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](Match& _) {
          return _(id = TypeParam)
            << _(id) << typevar(_, Type) << (Type << *_[rhs]);
        },

      In(TypeParams) * (!T(TypeParam))[TypeParam] >>
        [](Match& _) { return err(_[TypeParam], "expected a type parameter"); },

      // Params.
      T(Params) << T(List)[Params] >>
        [](Match& _) { return Params << *_[Params]; },

      // Param: (group ident type)
      In(Params) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = Param) << _(id) << typevar(_, Type) << DontCare;
        },

      // Param: (equals (group ident type) group)
      In(Params) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[Expr] * End) >>
        [](Match& _) {
          return _(id = Param)
            << _(id) << typevar(_, Type) << (Expr << *_[Expr]);
        },

      In(Params) * (!T(Param))[Param] >>
        [](Match& _) { return err(_[Param], "expected a parameter"); },

      // Use.
      (In(ClassBody) / In(FuncBody)) * T(Group)
          << T(Use)[Use] * (Any++)[Type] >>
        [](Match& _) {
          return (Use ^ _(Use)) << (Type << (_[Type] | DontCare));
        },

      T(Use)[Use] << End >>
        [](Match& _) { return err(_[Use], "can't put a `use` here"); },

      // TypeAlias: (group typealias ident typeparams type)
      (In(ClassBody) / In(FuncBody)) * T(Group)
          << (T(TypeAlias) * T(Ident)[id] * ~T(Square)[TypeParams] *
              ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = TypeAlias) << _(id) << (TypeParams << *_[TypeParams])
                                   << typevar(_, Type) << Type;
        },

      // TypeAlias: (equals (group typealias typeparams type) group)
      (In(ClassBody) / In(FuncBody)) * T(Equals)
          << ((T(Group)
               << (T(TypeAlias) * T(Ident)[id] * ~T(Square)[TypeParams] *
                   ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](Match& _) {
          return _(id = TypeAlias) << _(id) << (TypeParams << *_[TypeParams])
                                   << typevar(_, Type) << (Type << *_[rhs]);
        },

      (In(ClassBody) / In(FuncBody)) * T(TypeAlias)[TypeAlias] << End >>
        [](Match& _) {
          return err(_[TypeAlias], "expected a `type` definition");
        },
      T(TypeAlias)[TypeAlias] << End >>
        [](Match& _) {
          return err(_[TypeAlias], "can't put a `type` definition here");
        },

      // Class.
      (In(Top) / In(ClassBody) / In(FuncBody)) * T(Group)
          << (T(Class) * T(Ident)[id] * ~T(Square)[TypeParams] *
              ~T(Type)[Type] * T(Brace)[ClassBody] * (Any++)[rhs]) >>
        [](Match& _) {
          return Seq << (_(id = Class)
                         << _(id) << (TypeParams << *_[TypeParams])
                         << (_[Type] | Type) << (ClassBody << *_[ClassBody]))
                     << (Group << _[rhs]);
        },

      (In(Top) / In(ClassBody) / In(FuncBody)) * T(Class)[Class] << End >>
        [](Match& _) { return err(_[Class], "expected a `class` definition"); },
      T(Class)[Class] << End >>
        [](Match& _) {
          return err(_[Class], "can't put a `class` definition here");
        },

      // Type structure.
      TypeStruct * T(Group)[Type] >> [](Match& _) { return Type << *_[Type]; },
      TypeStruct * T(List)[TypeTuple] >>
        [](Match& _) { return TypeTuple << *_[TypeTuple]; },
      TypeStruct * T(Paren)[Type] >> [](Match& _) { return Type << *_[Type]; },

      // Lift anonymous structural types.
      TypeStruct * T(Brace)[ClassBody] >>
        [](Match& _) {
          auto id = _(ClassBody)->parent(ClassBody)->fresh();
          return Seq << (Lift
                         << ClassBody
                         << (_(id = TypeTrait)
                             << (Ident ^ id) << (ClassBody << *_[ClassBody])))
                     << (Ident ^ id);
        },

      // Allow `ref` to be used as a type name.
      TypeStruct * T(Ref) >> [](Match& _) { return Ident ^ ref; },

      TypeStruct *
          (T(Use) / T(Let) / T(Var) / T(Equals) / T(Class) / T(FatArrow) /
           T(TypeAlias) / T(Brace) / T(Ref) / Literal)[Type] >>
        [](Match& _) { return err(_[Type], "can't put this in a type"); },

      // A group can be in a FuncBody, ExprSeq, Tuple, or Assign.
      (In(FuncBody) / In(ExprSeq) / In(Tuple) / In(Assign)) * T(Group)[Group] >>
        [](Match& _) { return Expr << *_[Group]; },

      // An equals can be in a FuncBody, an ExprSeq, a Tuple, or an Expr.
      (In(FuncBody) / In(ExprSeq) / In(Tuple)) * T(Equals)[Equals] >>
        [](Match& _) { return Expr << (Assign << *_[Equals]); },
      In(Expr) * T(Equals)[Equals] >>
        [](Match& _) { return Assign << *_[Equals]; },

      // A list can be in a FuncBody, an ExprSeq, or an Expr.
      (In(FuncBody) / In(ExprSeq)) * T(List)[List] >>
        [](Match& _) { return Expr << (Tuple << *_[List]); },
      In(Expr) * T(List)[List] >> [](Match& _) { return Tuple << *_[List]; },

      // Empty parens are an empty Tuple.
      In(Expr) * (T(Paren) << End) >> ([](Match& _) -> Node { return Tuple; }),

      // Parens with one element are an Expr. Put the group, list, or equals
      // into the expr, where it will become an expr, tuple, or assign.
      In(Expr) * ((T(Paren) << (Any[lhs] * End))) >>
        [](Match& _) { return _(lhs); },

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

      // Lambda: (group typeparams) (list params...) => rhs
      In(Expr) * T(Brace)
          << (((T(Group) << T(Square)[TypeParams]) * T(List)[Params]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << *_[Params]) << (FuncBody << _[rhs]);
        },

      // Lambda: (group typeparams) (group param) => rhs
      In(Expr) * T(Brace)
          << (((T(Group) << T(Square)[TypeParams]) * T(Group)[Param]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << _[Param]) << (FuncBody << _[rhs]);
        },

      // Lambda: (list (group typeparams? param) params...) => rhs
      In(Expr) * T(Brace)
          << ((T(List)
               << ((T(Group) << (~T(Square)[TypeParams] * (Any++)[Param])) *
                   (Any++)[Params]))) *
            (T(Group) << T(FatArrow)) * (Any++)[rhs] >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (FuncBody << _[rhs]);
        },

      // Lambda: (group typeparams? param) => rhs
      In(Expr) * T(Brace)
          << ((T(Group) << (~T(Square)[TypeParams] * (Any++)[Param])) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (FuncBody << _[rhs]);
        },

      // Zero argument lambda.
      In(Expr) * T(Brace) << (!(T(Group) << T(FatArrow)))++[Lambda] >>
        [](Match& _) {
          return Lambda << TypeParams << Params << (FuncBody << _[Lambda]);
        },

      // Var.
      In(Expr) * T(Var)[Var] * T(Ident)[id] >>
        [](Match& _) { return _(id = Var) << _(id); },

      T(Var)[Var] << End >>
        [](Match& _) { return err(_[Var], "`var` needs an identifier"); },

      // Let.
      In(Expr) * T(Let)[Let] * T(Ident)[id] >>
        [](Match& _) { return _(id = Let) << _(id); },

      T(Let)[Let] << End >>
        [](Match& _) { return err(_[Let], "`let` needs an identifier"); },

      // Throw.
      In(Expr) * T(Throw) * Any[lhs] * (Any++)[rhs] >>
        [](Match& _) { return Throw << (Expr << _(lhs) << _[rhs]); },

      In(Expr) * T(Throw)[Throw] << End >>
        [](Match& _) { return err(_[Throw], "`throw` must specify a value"); },

      T(Throw)[Throw] << End >>
        [](Match& _) { return err(_[Throw], "can't put a `throw` here"); },

      // Move a ref to the last expr of a sequence.
      In(Expr) * T(Ref) * T(Expr)[Expr] >>
        [](Match& _) { return Expr << Ref << *_[Expr]; },
      In(Expr) * T(Ref) * T(Expr)[lhs] * T(Expr)[rhs] >>
        [](Match& _) { return Seq << _[lhs] << Ref << _[rhs]; },
      In(Expr) * T(Ref) * T(Expr)[Expr] * End >>
        [](Match& _) { return Expr << Ref << *_[Expr]; },

      // Lift Use, Class, TypeAlias to FuncBody.
      In(Expr) * (T(Use) / T(Class) / T(TypeAlias))[Lift] >>
        [](Match& _) { return Lift << FuncBody << _[Lift]; },

      // A Type at the end of an Expr is a TypeAssert. A tuple is never directly
      // wrapped in a TypeAssert, but an Expr containing a Tuple can be.
      T(Expr) << (((!T(Type))++)[Expr] * T(Type)[Type] * End) >>
        [](Match& _) {
          return Expr << (TypeAssert << _(Type) << (Expr << _[Expr]));
        },

      In(Expr) * (T(Package) / T(Iso) / T(Mut) / T(Imm) / T(FatArrow))[Expr] >>
        [](Match& _) {
          return err(_[Expr], "can't put this in an expression");
        },

      // Remove empty groups.
      T(Group) << End >> ([](Match& _) -> Node { return {}; }),
      T(Group)[Group] >> [](Match& _) { return err(_[Group], "syntax error"); },
    };
  }

  inline const auto TypeElem = T(Type) / T(TypeName) / T(TypeTuple) / T(Iso) /
    T(Imm) / T(Mut) / T(TypeList) / T(TypeView) / T(TypeFunc) / T(TypeThrow) /
    T(TypeIsect) / T(TypeUnion) / T(TypeVar) / T(TypeUnit) / T(Package);

  PassDef typeview()
  {
    return {
      TypeStruct * T(DontCare)[DontCare] >>
        [](Match& _) { return TypeVar ^ _.fresh(); },

      // Scoping binds most tightly.
      TypeStruct * T(Ident)[id] * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return TypeName << TypeUnit << _[id] << (_[TypeArgs] | TypeArgs);
        },
      TypeStruct * T(TypeName)[TypeName] * T(DoubleColon) * T(Ident)[id] *
          ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return TypeName << _[TypeName] << _[id] << (_[TypeArgs] | TypeArgs);
        },

      // Viewpoint adaptation binds more tightly than function types.
      TypeStruct * TypeElem[lhs] * T(Dot) * TypeElem[rhs] >>
        [](Match& _) {
          return TypeView << (Type << _[lhs]) << (Type << _[rhs]);
        },

      // TypeList binds more tightly than function types.
      TypeStruct * TypeElem[lhs] * T(Ellipsis) >>
        [](Match& _) { return TypeList << (Type << _[lhs]); },

      TypeStruct * T(DoubleColon)[DoubleColon] >>
        [](Match& _) { return err(_[DoubleColon], "misplaced type scope"); },
      TypeStruct * T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return err(_[TypeArgs], "type arguments on their own are not a type");
        },
      TypeStruct * T(Dot)[Dot] >>
        [](Match& _) { return err(_[Dot], "misplaced type viewpoint"); },
      TypeStruct * T(Ellipsis)[Ellipsis] >>
        [](Match& _) { return err(_[Ellipsis], "misplaced type list"); },
    };
  }

  PassDef typefunc()
  {
    return {
      // Function types bind more tightly than throw types. This is the only
      // right-associative operator.
      TypeStruct * TypeElem[lhs] * T(Symbol, "->") * TypeElem[rhs] *
          --T(Symbol, "->") >>
        [](Match& _) {
          return TypeFunc << (Type << _[lhs]) << (Type << _[rhs]);
        },
    };
  }

  PassDef typethrow()
  {
    return {
      // Throw types bind more tightly than isect and union types.
      TypeStruct * T(Throw) * TypeElem[rhs] >>
        [](Match& _) { return TypeThrow << (Type << _[rhs]); },
      TypeStruct * T(Throw)[Throw] >>
        [](Match& _) {
          return err(_[Throw], "must indicate what type is thrown");
        },
    };
  }

  PassDef typealg()
  {
    return {
      // Build algebraic types.
      TypeStruct * TypeElem[lhs] * T(Symbol, "&") * TypeElem[rhs] >>
        [](Match& _) {
          return TypeIsect << (Type << _[lhs]) << (Type << _[rhs]);
        },
      TypeStruct * TypeElem[lhs] * T(Symbol, "\\|") * TypeElem[rhs] >>
        [](Match& _) {
          return TypeUnion << (Type << _[lhs]) << (Type << _[rhs]);
        },

      TypeStruct * T(Symbol)[Symbol] >>
        [](Match& _) { return err(_[Symbol], "invalid symbol in type"); },
    };
  }

  PassDef typeflat()
  {
    return {
      // Flatten algebraic types.
      In(TypeUnion) * T(TypeUnion)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },
      In(TypeIsect) * T(TypeIsect)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },

      // Tuples of arity 1 are scalar types, tuples of arity 0 are the unit
      // type.
      T(TypeTuple) << (TypeElem[op] * End) >> [](Match& _) { return _(op); },
      T(TypeTuple) << End >> ([](Match& _) -> Node { return TypeUnit; }),

      // Flatten Type nodes. The top level Type node won't go away.
      TypeStruct * T(Type) << (TypeElem[op] * End) >>
        [](Match& _) { return _(op); },

      // Empty types are the unit type.
      T(Type)[Type] << End >> [](Match& _) { return Type << TypeUnit; },

      In(TypeThrow) * T(TypeThrow)[lhs] >>
        [](Match& _) { return err(_[lhs], "can't throw a throw type"); },

      T(Type)[Type] << (Any * Any) >>
        [](Match& _) {
          return err(_[Type], "can't use adjacency to specify a type");
        },
    };
  }

  PassDef typednf()
  {
    return {
      // throw (A | B) -> throw A | throw B
      T(TypeThrow) << T(TypeUnion)[op] >>
        [](Match& _) {
          Node r = TypeUnion;
          for (auto& child : *_(op))
            r << (TypeThrow << child);
          return r;
        },

      // (A | B) & C -> (A & C) | (B & C)
      T(TypeIsect)
          << (((!T(TypeUnion))++)[lhs] * T(TypeUnion)[op] * (Any++)[rhs]) >>
        [](Match& _) {
          Node r = TypeUnion;
          for (auto& child : *_(op))
            r << (TypeIsect << clone(_[lhs]) << clone(child) << clone(_[rhs]));
          return r;
        },

      // Re-flatten algebraic types, as DNF can produce them.
      In(TypeUnion) * T(TypeUnion)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },
      In(TypeIsect) * T(TypeIsect)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },

      // (throw A) & (throw B) -> throw (A & B)
      T(TypeIsect) << ((T(TypeThrow)++)[op] * End) >>
        [](Match& _) {
          Node r = TypeIsect;
          auto& end = _[op].second;
          for (auto& it = _[op].first; it != end; ++it)
            r << (*it)->front();
          return TypeThrow << r;
        },

      // (throw A) & B -> invalid
      In(TypeIsect) * T(TypeThrow)[op] >>
        [](Match& _) {
          return err(
            _[op], "can't intersect a throw type with a non-throw type");
        },

      // Re-check as these can be generated by DNF.
      In(TypeThrow) * T(TypeThrow)[lhs] >>
        [](Match& _) { return err(_[lhs], "can't throw a throw type"); },
    };
  }

  PassDef include()
  {
    // This needs the symbol tables for classes to be built.
    // TODO: rebuilding symbol tables requires rebuilding includes
    // don't need Include vs Use, as failure becomes Error?
    // make it a predicate, only rewrite on failure?
    return {
      T(Use)[lhs] << (T(TypeName)[rhs] * End) >>
        [](Match& _) {
          auto site = Include ^ _(lhs);
          auto found = resolve(_(rhs));

          if (found.def)
          {
            _.include(site, found.def);
            return site << _[rhs];
          }

          return err(_[lhs], "couldn't resolve this type");
        },

      T(Use)[Use] << (!T(TypeName) * End) >>
        [](Match& _) { return err(_[Use], "`use` requires a type name"); },
    };
  }

  PassDef reference()
  {
    return {
      // Dot notation. Don't interpret `id` as a local variable.
      In(Expr) * T(Dot) * Name[id] * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return Seq << Dot << (Selector << _[id] << (_[TypeArgs] | TypeArgs));
        },

      // Local reference.
      In(Expr) * Name[id]([](auto& n) { return lookup(n, {Var}); }) >>
        [](Match& _) { return RefVar << _(id); },

      In(Expr) * T(Ident)[id]([](auto& n) {
        return lookup(n, {Let, Param});
      }) >>
        [](Match& _) { return RefLet << _(id); },

      // Unscoped type reference.
      In(Expr) * T(Ident)[id]([](auto& n) {
        return lookup(n, {Class, TypeAlias, TypeTrait, TypeParam});
      }) * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return TypeName << TypeUnit << _(id) << (_[TypeArgs] | TypeArgs);
        },

      // Unscoped reference that isn't a local or a type. Treat it as a
      // selector, even if it resolves to a Function.
      In(Expr) * Name[id] * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) { return Selector << _(id) << (_[TypeArgs] | TypeArgs); },

      // Scoped lookup.
      In(Expr) *
          (T(TypeName)[lhs] * T(DoubleColon) * Name[id] *
           ~T(TypeArgs)[TypeArgs])[Type] >>
        [](Match& _) {
          auto found = resolve(_(lhs));
          auto def = lookdown(found, _(id));

          if (!def)
            return err(_[Type], "couldn't resolve this scoped name");

          if (def->type().in({Class, TypeAlias, TypeTrait, TypeParam}))
            return TypeName << _[lhs] << _(id) << (_[TypeArgs] | TypeArgs);

          return FunctionName << _[lhs] << _(id) << (_[TypeArgs] | TypeArgs);
        },

      In(Expr) * T(DoubleColon) >>
        [](Match& _) { return err(_[DoubleColon], "expected a scoped name"); },

      // Create sugar.
      In(Expr) * T(TypeName)[lhs] * ~T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return Call
            << (FunctionName << _[lhs] << (Ident ^ create)
                             << (_[TypeArgs] | TypeArgs));
        },

      // Lone TypeArgs are typeargs on apply.
      In(Expr) * T(TypeArgs)[TypeArgs] >>
        [](Match& _) {
          return Seq << Dot << (Selector << (Ident ^ apply) << _[TypeArgs]);
        },

      // TypeAssert on a Selector or FunctionName.
      T(TypeAssert)
          << (T(Type)[lhs] *
              (T(Expr) << (T(Selector) / T(FunctionName))[rhs])) >>
        [](Match& _) { return TypeAssertOp << _[lhs] << _[rhs]; },

      // Compact expressions.
      In(Expr) * T(Expr) << (Any[Expr] * End) >>
        [](Match& _) { return _(Expr); },
      T(Expr) << (T(Expr)[Expr] * End) >> [](Match& _) { return _(Expr); },
    };
  }

  inline const auto Object = Literal / T(RefVar) / T(RefVarLHS) / T(RefLet) /
    T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) / T(Assign) / T(Expr) /
    T(ExprSeq) / T(TypeAssert);
  inline const auto Operator = T(FunctionName) / T(Selector) / T(TypeAssertOp);
  inline const auto Apply = (Selector << (Ident ^ apply) << TypeArgs);

  auto arg(Node args, Node arg)
  {
    if (arg)
    {
      if (arg->type() == Tuple)
        args->push_back({arg->begin(), arg->end()});
      else if (arg->type() == Expr)
        args << arg;
      else
        args << (Expr << arg);
    }

    return args;
  }

  auto call(Node op, Node lhs, Node rhs = {})
  {
    return Call << op << arg(arg(Args, lhs), rhs);
  }

  PassDef reverseapp()
  {
    return {
      // Dot: reverse application. This binds most strongly.
      (Object / Operator)[lhs] * T(Dot) * Operator[rhs] >>
        [](Match& _) { return call(_(rhs), _(lhs)); },

      (Object / Operator)[lhs] * T(Dot) * (T(Tuple) / Object)[rhs] >>
        [](Match& _) { return call(clone(Apply), _(rhs), _(lhs)); },

      T(Dot)[Dot] >>
        [](Match& _) {
          return err(_[Dot], "must use `.` with values and operators");
        },
    };
  }

  PassDef application()
  {
    // These rules allow expressions such as `3 * -4` or `a and not b` to have
    // the expected meaning.
    return {
      // Adjacency: application.
      In(Expr) * Object[lhs] * Object[rhs] >>
        [](Match& _) { return call(clone(Apply), _(lhs), _(rhs)); },

      // Prefix. This doesn't rewrite `op op`.
      In(Expr) * Operator[op] * Object[rhs] >>
        [](Match& _) { return call(_(op), _(rhs)); },

      // Infix. This doesn't rewrite with an operator on lhs or rhs.
      In(Expr) * Object[lhs] * Operator[op] * Object[rhs] >>
        [](Match& _) { return call(_(op), _(lhs), _(rhs)); },

      // Postfix. This doesn't rewrite unless only postfix operators remain.
      In(Expr) * (Object / Operator)[lhs] * Operator[op] * Operator++[rhs] *
          End >>
        [](Match& _) { return Seq << call(_(op), _(lhs)) << _[rhs]; },

      // Ref expressions.
      T(Ref) * T(RefVar)[RefVar] >>
        [](Match& _) { return RefVarLHS ^ _(RefVar); },
      T(Ref) * T(Call)[Call] >> [](Match& _) { return CallLHS << *_[Call]; },

      // Tuple flattening.
      In(Tuple) * T(Expr) << (Object[lhs] * T(Ellipsis) * End) >>
        [](Match& _) { return Expr << (TupleFlatten << (Expr << _(lhs))); },

      // Use DontCare for partial application of arbitrary arguments.
      T(Call)
          << (Operator[op] *
              (T(Args)
               << ((T(Expr) << !T(DontCare))++ * (T(Expr) << T(DontCare)) *
                   T(Expr)++))[Args]) >>
        [](Match& _) {
          Node params = Params;
          Node args = Args;
          auto lambda = Lambda
            << TypeParams << params
            << (FuncBody << (Expr << (Call << _(op) << args)));

          for (auto& arg : *_(Args))
          {
            if (arg->front()->type() == DontCare)
            {
              auto id = _.fresh();
              params
                << (_(id = Param) << (Ident ^ id) << typevar(_) << DontCare);
              args << (Expr << (RefLet ^ id));
            }
            else
            {
              args << arg;
            }
          }

          return lambda;
        },

      T(Ref) >>
        [](Match& _) {
          return err(_[Ref], "must use `ref` in front of a variable or call");
        },

      T(Ellipsis) >>
        [](Match& _) {
          return err(_[Ellipsis], "must use `...` after a value in a tuple");
        },

      In(Expr) * T(DontCare) >>
        [](Match& _) {
          return err(_[DontCare], "must use `_` in a partial application");
        },

      In(Expr) * (Any * Any)[Expr] >>
        [](Match& _) {
          return err(_[Expr], "adjacency on this expression isn't meaningful");
        },
    };
  }

  PassDef assignlhs()
  {
    return {
      // Turn a Tuple on the LHS of an assignment into a TupleLHS.
      (In(Assign) / In(TupleLHS)) * (T(Expr) << T(Tuple)[lhs]) >>
        [](Match& _) { return Expr << (TupleLHS << *_[lhs]); },

      (In(Assign) / In(TupleLHS)) *
          (T(Expr) << (T(TypeAssert) << (T(Type)[Type] * T(Tuple)[lhs]))) >>
        [](Match& _) {
          return Expr << (TypeAssert << _(Type) << (TupleLHS << *_[lhs]));
        },

      // Turn a Call on the LHS of an assignment into a CallLHS.
      (In(Assign) / In(TupleLHS)) * (T(Expr) << T(Call)[lhs]) >>
        [](Match& _) { return Expr << (CallLHS << *_[lhs]); },

      (In(Assign) / In(TupleLHS)) *
          (T(Expr) << (T(TypeAssert) << (T(Type)[Type] * T(Call)[lhs]))) >>
        [](Match& _) {
          return Expr << (TypeAssert << _(Type) << (CallLHS << *_[lhs]));
        },

      // Turn a RefVar on the LHS of an assignment into a RefVarLHS.
      (In(Assign) / In(TupleLHS)) * (T(Expr) << T(RefVar)[lhs]) >>
        [](Match& _) { return Expr << (RefVarLHS ^ _(lhs)); },

      (In(Assign) / In(TupleLHS)) *
          (T(Expr) << (T(TypeAssert) << (T(Type)[Type] * T(RefVar)[lhs]))) >>
        [](Match& _) {
          return Expr << (TypeAssert << _(Type) << (RefVarLHS << *_[lhs]));
        },
    };
  }

  inline const auto Std = TypeName << TypeUnit << (Ident ^ standard)
                                   << TypeArgs;
  inline const auto Cell = TypeName << Std << (Ident ^ cell) << TypeArgs;
  inline const auto CellCreate =
    (FunctionName << Cell << (Ident ^ create) << TypeArgs);
  inline const auto CallCellCreate = (Call << CellCreate);
  inline const auto Load = (Selector << (Ident ^ load) << TypeArgs);

  PassDef localvar()
  {
    return {
      T(Var)[Var] >>
        [](Match& _) {
          auto id = _(Var)->at(wf / Var / Ident)->location();
          return ExprSeq << (Expr
                             << (Assign << (Expr << (_(id = Let)))
                                        << (Expr << CallCellCreate)))
                         << (Expr << (RefLet ^ id));
        },

      T(RefVar)[RefVar] >>
        [](Match& _) { return Call << clone(Load) << (RefLet ^ _(RefVar)); },

      T(RefVarLHS)[RefVarLHS] >> [](Match& _) { return RefLet ^ _(RefVarLHS); },
    };
  }

  inline const auto Store = (Selector << (Ident ^ store) << TypeArgs);

  PassDef assignment()
  {
    return {
      // Let binding.
      In(Assign) *
          (T(Expr)
           << ((T(Let) << T(Ident)[id]) /
               (T(TypeAssert) << T(Type)[Type] << (T(Let) << T(Ident)[id])))) *
          T(Expr)[rhs] * End >>
        [](Match& _) {
          return ExprSeq << (Expr
                             << (_(id = Bind)
                                 << id << typevar(_, Type) << _(rhs)))
                         << (Expr << (RefLet ^ id));
        },

      // Destructuring assignment.
      In(Assign) *
          (T(Expr)
           << (T(TupleLHS)[lhs] /
               (T(TypeAssert)
                << (T(Type)[Type] * (T(Expr) << T(TupleLHS)[lhs]))))) *
          T(Expr)[rhs] * End >>
        [](Match& _) {
          // let $id = rhs
          auto id = _.fresh();
          auto seq = ExprSeq
            << (Expr << (Assign << (Expr << (_(id = Let))) << _(rhs)));

          Node lhs_tuple = Tuple;
          Node rhs_tuple = Tuple;
          auto ty = _(Type);
          size_t index = 0;

          for (auto child : *_(lhs))
          {
            // let $tid = tuple_child
            auto tid = _.fresh();
            seq
              << (Expr
                  << (Assign << (Expr << (Let << (Ident ^ tid))) << child));

            // Build a LHS tuple that will only be used if there's a TypeAssert.
            if (ty)
              lhs_tuple << (Expr << (RefLet ^ tid));

            // $tid = $id._index
            rhs_tuple
              << (Expr
                  << (Assign
                      << (RefLet ^ tid)
                      << (Expr
                          << (Call
                              << (Selector
                                  << (Ident ^
                                      Location("_" + std::to_string(index++)))
                                  << TypeArgs)
                              << (RefLet ^ id)))));
          }

          // TypeAssert comes after the let bindings for the LHS.
          if (ty)
            seq << (Expr << (TypeAssert << ty << lhs_tuple));

          // The RHS tuple is the last expression in the sequence.
          return Expr << (seq << (Expr << rhs_tuple));
        },

      // Assignment to anything else.
      In(Assign) * T(Expr)[lhs] * T(Expr)[rhs] * End >>
        [](Match& _) {
          return Expr << (Call << clone(Store) << (Args << _(lhs) << _(rhs)));
        },

      // Compact assigns after they're reduced.
      T(Assign) << ((T(Expr) << Any[lhs]) * End) >>
        [](Match& _) { return _(lhs); },

      T(Let)[Let] >>
        [](Match& _) { return err(_[Let], "must assign to a `let` binding"); },
    };
  }

  inline const auto Liftable = T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) /
    T(Selector) / T(FunctionName) / Literal / T(Throw);

  PassDef anf()
  {
    // TODO: do DontCare in an earlier pass, if it's in Args, make a lambda
    return {
      // This liftable expr is already bound from `let x = e`.
      In(Bind) * (T(Expr) << Liftable[Lift]) >>
        [](Match& _) { return _(Lift); },

      In(Bind) * (T(Expr) << T(Bind)[Bind]) >>
        [](Match& _) {
          return err(
            _[Bind],
            "well-formedness allows this but it can't occur on written code");
        },

      // Lift `let x` bindings, leaving the RefLet behind.
      T(Expr) << T(Bind)[Bind] >>
        [](Match& _) { return Lift << FuncBody << _(Bind); },

      // Lift RefLet by one step everywhere.
      T(Expr) << T(RefLet)[RefLet] >> [](Match& _) { return _(RefLet); },

      // Create a new binding for this liftable expr.
      T(Expr)
          << (Liftable[Lift] /
              ((T(TypeAssert) / T(TypeAssertOp))
               << (T(Type)[Type] * Liftable[Lift]))) >>
        [](Match& _) {
          auto id = _.fresh();
          return Seq << (Lift << FuncBody
                              << (_(id = Bind) << (Ident ^ id)
                                               << typevar(_, Type) << _(Lift)))
                     << (RefLet ^ id);
        },

      // Compact an ExprSeq with only one element.
      T(ExprSeq) << (Any[lhs] * End) >> [](Match& _) { return _(lhs); },

      // Discard leading RefLets in ExprSeq.
      In(ExprSeq) * (T(RefLet) * Any[lhs] * Any++[rhs]) >>
        [](Match& _) { return Seq << _(lhs) << _[rhs]; },

      // Tuple flattening.
      In(Tuple) * (T(Expr) << T(TupleFlatten)[TupleFlatten]) * End >>
        [](Match& _) { return _(TupleFlatten); },
      T(TupleFlatten)[TupleFlatten] >>
        [](Match& _) {
          return err(_[TupleFlatten], "`...` can only appear in tuples");
        },
    };
  }

  Driver& driver()
  {
    static Driver d(
      "Verona",
      parser(),
      wfParser(),
      {
        {"modules", modules(), wfPassModules()},
        {"structure", structure(), wfPassStructure()},
        {"typeview", typeview(), wfPassTypeView()},
        {"typefunc", typefunc(), wfPassTypeFunc()},
        {"typethrow", typethrow(), wfPassTypeThrow()},
        {"typealg", typealg(), wfPassTypeAlg()},
        {"typeflat", typeflat(), wfPassTypeFlat()},
        {"typednf", typednf(), wfPassTypeDNF()},
        {"include", include(), wfPassInclude()},
        {"reference", reference(), wfPassReference()},
        {"reverseapp", reverseapp(), wfPassReverseApp()},
        {"application", application(), wfPassApplication()},
        {"assignlhs", assignlhs(), wfPassAssignLHS()},
        {"localvar", localvar(), wfPassLocalVar()},
        {"assignment", assignment(), wfPassAssignment()},
        {"anf", anf(), wfPassANF()},
      });

    return d;
  }
}
