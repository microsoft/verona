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

  inline const auto ExprStruct =
    In(FuncBody) / In(Assign) / In(Tuple) / In(Expr);
  inline const auto TypeStruct = In(Type) / In(TypeList) / In(TypeTuple) /
    In(TypeView) / In(TypeFunc) / In(TypeThrow) / In(TypeUnion) / In(TypeIsect);
  inline const auto Name = T(Ident) / T(Symbol);
  inline const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) /
    T(Hex) / T(Bin) / T(Int) / T(Float) / T(HexFloat);

  auto typevar(auto& _, const Token& t)
  {
    auto n = _(t);
    return n ? n : Type << (TypeVar ^ _.fresh());
  }

  auto letbind(auto& _, Location& id, Node t, Node e)
  {
    return (Lift << FuncBody << (_(id = Let) << (Ident ^ id) << t << e));
  }

  PassDef structure()
  {
    return {
      // Let Field:
      // (equals (group let ident type) group)
      // (group let ident type)
      In(ClassBody) *
          ((T(Equals)
            << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
                T(Group)[rhs] * End)) /
           (T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End))) >>
        [](Match& _) {
          return _(id = FieldLet)
            << _(id) << typevar(_, Type) << (Expr << *_[rhs]);
        },

      // Var Field:
      // (equals (group var ident type) group)
      // (group var ident type)
      In(ClassBody) *
          ((T(Equals)
            << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
                T(Group)[rhs] * End)) /
           (T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End))) >>
        [](Match& _) {
          return _(id = FieldVar)
            << _(id) << typevar(_, Type) << (Expr << *_[rhs]);
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
          return _(id = Param) << _(id) << typevar(_, Type) << Expr;
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

      TypeStruct *
          (T(Use) / T(Let) / T(Var) / T(Equals) / T(Class) / T(FatArrow) /
           T(TypeAlias) / T(Brace) / T(Ref) / Literal)[Type] >>
        [](Match& _) { return err(_[Type], "can't put this in a type"); },

      // Expression structure.
      ExprStruct * T(Group)[Expr] >> [](Match& _) { return Expr << *_[Expr]; },
      ExprStruct * T(List)[Tuple] >>
        [](Match& _) { return Tuple << *_[Tuple]; },
      ExprStruct * T(Equals)[Assign] >>
        [](Match& _) { return Assign << *_[Assign]; },

      // Empty parens are an empty tuple.
      ExprStruct * T(Paren) << End >> ([](Match& _) -> Node { return Tuple; }),
      ExprStruct * T(Paren)[Expr] >> [](Match& _) { return Expr << *_[Expr]; },

      // Typearg structure.
      (TypeStruct / ExprStruct) * T(Square)[TypeArgs] >>
        [](Match& _) { return TypeArgs << *_[TypeArgs]; },
      T(TypeArgs) << T(List)[TypeArgs] >>
        [](Match& _) { return TypeArgs << *_[TypeArgs]; },
      In(TypeArgs) * T(Group)[Type] >>
        [](Match& _) { return Type << *_[Type]; },
      In(TypeArgs) * T(Paren)[Type] >>
        [](Match& _) { return Type << *_[Type]; },

      // Lambda: (group typeparams) (list params...) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[TypeParams]) * T(List)[Params]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << *_[Params]) << (FuncBody << _[rhs]);
        },

      // Lambda: (group typeparams) (group param) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[TypeParams]) * T(Group)[Param]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << _[Param]) << (FuncBody << _[rhs]);
        },

      // Lambda: (list (group typeparams? param) params...) => rhs
      ExprStruct * T(Brace)
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
      ExprStruct * T(Brace)
          << ((T(Group) << (~T(Square)[TypeParams] * (Any++)[Param])) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (TypeParams << *_[TypeParams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (FuncBody << _[rhs]);
        },

      // Zero argument lambda.
      ExprStruct * T(Brace) << (!(T(Group) << T(FatArrow)))++[Lambda] >>
        [](Match& _) {
          return Lambda << TypeParams << Params << (FuncBody << _[Lambda]);
        },

      // Var.
      ExprStruct * T(Var)[Var] * T(Ident)[id] >>
        [](Match& _) { return _(id = Var) << _(id); },

      T(Var)[Var] << End >>
        [](Match& _) { return err(_[Var], "`var` needs an identifier"); },

      // Let.
      ExprStruct * T(Let)[Let] * T(Ident)[id] >>
        [](Match& _) { return _(id = Let) << _(id); },

      T(Let)[Let] << End >>
        [](Match& _) { return err(_[Let], "`let` needs an identifier"); },

      // Throw.
      ExprStruct * T(Throw) * (Any++)[rhs] >>
        [](Match& _) { return Throw << (Expr << _[rhs]); },

      T(Throw)[Throw] << End >>
        [](Match& _) { return err(_[Throw], "can't put a `throw` here"); },

      // Move a ref to the last expr of a sequence.
      ExprStruct * T(Ref) * T(Expr)[Expr] >>
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

      ExprStruct *
          (T(Package) / T(Iso) / T(Mut) / T(Imm) / T(FatArrow))[Expr] >>
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

      // Any Use nodes that remain are ill-formed.
      T(Use)[Use] >>
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
        return lookup(n, {Class, TypeAlias, TypeParam});
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

          if (def->type().in({Class, TypeAlias, TypeParam}))
            return TypeName << _[lhs] << _(id) << (_[TypeArgs] | TypeArgs);

          return FunctionName << _[lhs] << _(id) << (_[TypeArgs] | TypeArgs);
        },

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

      // Compact expressions.
      In(Expr) * T(Expr) << (Any[Expr] * End) >>
        [](Match& _) { return _(Expr); },
      T(Expr) << (T(Expr)[Expr] * End) >> [](Match& _) { return _(Expr); },
    };
  }

  inline const auto Object0 = Literal / T(RefVar) / T(RefVarLHS) / T(RefLet) /
    T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) / T(Assign) / T(Expr);
  inline const auto Object = Object0 / (T(TypeAssert) << (T(Type) * Object0));
  inline const auto Operator0 = T(FunctionName) / T(Selector);
  inline const auto Operator =
    Operator0 / (T(TypeAssert) << (T(Type) * Operator0));

  inline const auto Apply = (Selector << (Ident ^ apply) << TypeArgs);
  inline const auto Load = (Selector << (Ident ^ load) << TypeArgs);
  inline const auto Store = (Selector << (Ident ^ store) << TypeArgs);
  inline const auto Std = TypeName << TypeUnit << (Ident ^ standard)
                                   << TypeArgs;
  inline const auto Cell = TypeName << Std << (Ident ^ cell) << TypeArgs;
  inline const auto CellCreate =
    (FunctionName << Cell << (Ident ^ create) << TypeArgs);
  inline const auto CallCellCreate = (Call << CellCreate);

  PassDef reverseapp()
  {
    return {
      // Dot: reverse application. This binds most strongly.
      // TODO: rhs could be a TypeAssert, extract it in Call
      In(Expr) * T(Tuple)[lhs] * T(Dot) * Operator[rhs] >>
        [](Match& _) { return Call << _(rhs) << (Args << *_[lhs]); },
      In(Expr) * (Object / Operator)[lhs] * T(Dot) * Operator[rhs] >>
        [](Match& _) { return Call << _(rhs) << (Args << _[lhs]); },

      In(Expr) * T(Tuple)[lhs] * T(Dot) * Object[rhs] >>
        [](Match& _) {
          return Call << clone(Apply) << (Args << _[rhs] << *_[lhs]);
        },
      In(Expr) * (Object / Operator)[lhs] * T(Dot) * Object[rhs] >>
        [](Match& _) {
          return Call << clone(Apply) << (Args << _[rhs] << _[lhs]);
        },

      In(Expr) * T(Tuple)[lhs] * T(Dot) * T(Tuple)[rhs] >>
        [](Match& _) {
          return Call << clone(Apply) << (Args << *_[rhs] << *_[lhs]);
        },
      In(Expr) * (Object / Operator)[lhs] * T(Dot) * T(Tuple)[rhs] >>
        [](Match& _) {
          return Call << clone(Apply) << (Args << *_[rhs] << _[lhs]);
        },

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
      In(Expr) * T(Tuple)[lhs] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << clone(Apply) << *_[lhs] << *_[rhs]; },
      In(Expr) * T(Tuple)[lhs] * Object[rhs] >>
        [](Match& _) { return Call << clone(Apply) << *_[lhs] << _[rhs]; },
      In(Expr) * Object[lhs] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[lhs] << *_[rhs]; },
      In(Expr) * Object[lhs] * Object[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[lhs] << _[rhs]; },

      // TODO: op could be a TypeAssert, extract it in Call
      // Prefix. This doesn't rewrite `op op`.
      In(Expr) * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << *_[rhs]; },
      In(Expr) * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << _[rhs]; },

      // Infix. This doesn't rewrite with an operator on lhs or rhs.
      In(Expr) * T(Tuple)[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << *_[lhs] << *_[rhs]; },
      In(Expr) * T(Tuple)[lhs] * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << *_[lhs] << _[rhs]; },
      In(Expr) * Object[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << _[lhs] << *_[rhs]; },
      In(Expr) * Object[lhs] * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << _[lhs] << _[rhs]; },

      // Postfix. This doesn't rewrite unless the expression ends here.
      // TODO: sequence of postfix operators?
      In(Expr) * T(Tuple)[lhs] * Operator[op] * End >>
        [](Match& _) { return Call << _[op] << *_[lhs]; },
      In(Expr) * (Object / Operator)[lhs] * Operator[op] * End >>
        [](Match& _) { return Call << _[op] << _[lhs]; },

      // Ref expressions.
      In(Expr) * T(Ref) * T(RefVar)[RefVar] >>
        [](Match& _) { return RefVarLHS ^ _(RefVar); },
      In(Expr) * T(Ref) * T(Call)[Call] >>
        [](Match& _) { return CallLHS << *_[Call]; },

      T(Ref) >>
        [](Match& _) {
          return err(_[Ref], "must use `ref` in front of a variable or call");
        },

      // TODO: remaining Operators are errors? rules on let, var, throw,
      // dontcare, ellipsis?
    };
  }

  PassDef assignlhs()
  {
    return {
      // Turn a Tuple on the LHS of an assignment into a TupleLHS.
      In(Assign) * (T(Expr) << (T(Tuple)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](Match& _) {
          return Seq << (Expr << (TupleLHS << *_[lhs]) << _[Type]) << _[rhs];
        },

      // Turn a Call on the LHS of an assignment into a CallLHS.
      In(Assign) * (T(Expr) << (T(Call)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](Match& _) {
          return Seq << (Expr << (CallLHS << *_[lhs]) << _[Type]) << _[rhs];
        },

      // Turn a RefVar on the LHS of an assignment into a RefVarLHS.
      In(Assign) * (T(Expr) << (T(RefVar)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](Match& _) {
          return Seq << (Expr << (RefVarLHS ^ _(lhs)) << _[Type]) << _[rhs];
        },

      // Recurse LHS.
      In(TupleLHS) * (T(Expr) << (T(Tuple)[lhs] * ~T(Type)[Type])) >>
        [](Match& _) { return Expr << (TupleLHS << *_[lhs]) << _[Type]; },

      In(TupleLHS) * (T(Expr) << (T(Call)[lhs] * ~T(Type)[Type])) >>
        [](Match& _) { return Expr << (CallLHS << *_[lhs]) << _[Type]; },

      In(TupleLHS) * (T(Expr) << (T(RefVar)[lhs] * ~T(Type)[Type])) >>
        [](Match& _) { return Expr << (RefVarLHS ^ _(lhs)) << _[Type]; },
    };
  }

  inline const auto InContainer =
    In(Expr) / In(Tuple) / In(Call) / In(CallLHS) / In(Assign);

  PassDef localvar()
  {
    return {
      (In(FuncBody) / InContainer) * T(Var)[Var] >>
        [](Match& _) {
          auto var = _(Var);
          auto id = var->at(wf / Var / Ident)->location();
          auto t = var->at(wf / Var / Type);
          return Seq << letbind(_, id, t, CallCellCreate) << (RefLet ^ id);
        },

      (In(FuncBody) / InContainer) * T(RefVar)[RefVar] >>
        [](Match& _) { return Call << clone(Load) << (RefLet ^ _(RefVar)); },

      (In(FuncBody) / InContainer) * T(RefVarLHS)[RefVarLHS] >>
        [](Match& _) { return RefLet ^ _(RefVarLHS); },
    };
  }

  PassDef destructure()
  {
    return {
      // Destructuring assignment.
      In(Assign) * (T(Expr) << (T(TupleLHS)[lhs] * ~T(Type)[ltype] * End)) *
          (T(Expr) << (Any[rhs] * ~T(Type)[rtype] * End)) * End >>
        [](Match& _) {
          auto e = _(rhs);
          auto id = _.fresh();
          Node tuple = Tuple;
          size_t index = 0;

          for (auto child : *_(lhs))
          {
            tuple
              << (Assign << child
                         << (Expr
                             << (Call << (Selector
                                          << (Ident ^
                                              Location(
                                                "_" + std::to_string(index++)))
                                          << TypeArgs)
                                      << (RefLet ^ id))));
          }

          return Seq << letbind(_, id, typevar(_, rtype), e)
                     << (Expr << tuple << _[ltype]);
        },
    };
  }

  inline const auto LiftExpr = T(Tuple) / T(Lambda) / T(Call) / T(CallLHS);
  inline const auto RHSExpr = T(RefLet) / Literal;

  PassDef anf()
  {
    return {
      // Lift an expression as a let with a type assertion.
      (In(FuncBody) / InContainer) * LiftExpr[Lift] * ~T(Type)[Type] >>
        [](Match& _) {
          auto id = _.fresh();
          return Seq << letbind(_, id, typevar(_, Type), _(Lift))
                     << (RefLet ^ id);
        },

      // Lift type assertions on RefLet and literals as new lets.
      (In(FuncBody) / InContainer) * RHSExpr[Lift] * T(Type)[Type] >>
        [](Match& _) {
          auto id = _.fresh();
          return Seq << letbind(_, id, _(Type), _(Lift)) << (RefLet ^ id);
        },

      // Compact exprs after they're reduced.
      T(Expr) << (Any[op] * End) >> [](Match& _) { return _(op); },
    };
  }

  PassDef assignment()
  {
    return {
      // Let binding.
      In(Assign) * T(Let)[lhs] * RHSExpr[rhs] * End >>
        [](Match& _) {
          return Seq << (Lift << FuncBody << (_(lhs) << _(rhs)))
                     << (RefLet ^ _(lhs)->at(wf / Let / Ident));
        },

      // Assignment to RefLet.
      In(Assign) * T(RefLet)[lhs] * RHSExpr[rhs] * End >>
        [](Match& _) {
          auto id = _.fresh();
          auto e = Call << clone(Store) << _(lhs) << _(rhs);
          return Seq << letbind(_, id, TypeVar ^ _.fresh(), e) << (RefLet ^ id);
        },

      // Compact assigns after they're reduced.
      T(Assign) << (Any[op] * End) >> [](Match& _) { return _(op); },
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
        {"assignlhs", assignlhs(), {}},
        {"localvar", localvar(), {}},
        {"destructure", destructure(), {}},
        {"anf", anf(), {}},
        {"assignment", assignment(), {}},
        {"infer", infer(), {}},
      });

    return d;
  }
}
