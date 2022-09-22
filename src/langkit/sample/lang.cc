// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

#include "wf.h"

namespace sample
{
  PassDef modules()
  {
    return {
      // Module.
      T(Directory)[Directory] << (T(File)++)[File] >>
        [](Match& _) {
          auto ident = path::last(_(Directory)->location().source->origin());
          return Group << (Class ^ _(Directory)) << (Ident ^ ident)
                       << (Brace << *_[File]);
        },

      // File on its own (no module).
      In(Top) * T(File)[File] >>
        [](Match& _) {
          auto ident = path::last(_(File)->location().source->origin());
          return Group << (Class ^ _(File)) << (Ident ^ ident)
                       << (Brace << *_[File]);
        },

      // Allow disambiguating anonymous types from class or function bodies.
      In(Group) * T(Typealias) * T(Brace)[Classbody] >>
        [](Match& _) { return TypeTrait << (Classbody << *_[Classbody]); },
    };
  }

  PassDef types()
  {
    return {
      // Packages.
      T(Package) * (T(String) / T(Escaped))[String] >>
        [](Match& _) { return Package << _[String]; },

      // Type.
      T(Colon)[Colon] * ((!T(Brace))++)[Type] >>
        [](Match& _) { return (Type ^ _(Colon)) << _[Type]; },
    };
  }

  inline const auto ExprStruct =
    In(Funcbody) / In(Assign) / In(Tuple) / In(Expr);
  inline const auto TypeStruct = In(Type) / In(TypeTuple);
  inline const auto SeqStruct = T(Expr) / T(Tuple) / T(Assign);
  inline const auto Name = T(Ident) / T(Symbol);
  inline const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) /
    T(Hex) / T(Bin) / T(Int) / T(Float) / T(HexFloat);

  auto typevar(auto& _, const Token& e, const Token& t)
  {
    return _(t) ? _(t) : TypeVar ^ _(e)->fresh();
  }

  auto letbind(auto& _, Location& id, Node t, Node e)
  {
    return (Lift << Funcbody << (_(id = Let) << (Ident ^ id) << t << e));
  }

  PassDef structure()
  {
    return {
      // Let Field:
      // (equals (group let ident type) group)
      // (group let ident type)
      In(Classbody) *
          ((T(Equals)
            << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
                T(Group)[rhs] * End)) /
           (T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End))) >>
        [](Match& _) {
          return _(id = FieldLet)
            << _(id) << typevar(_, id, Type) << (Expr << *_[rhs]);
        },

      // Var Field:
      // (equals (group var ident type) group)
      // (group var ident type)
      In(Classbody) *
          ((T(Equals)
            << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
                T(Group)[rhs] * End)) /
           (T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End))) >>
        [](Match& _) {
          return _(id = FieldVar)
            << _(id) << typevar(_, id, Type) << (Expr << *_[rhs]);
        },

      // Function.
      // (equals (group name square parens type) group)
      In(Classbody) *
          (T(Equals)
           << (T(Group) << (~Name[id] * ~T(Square)[Typeparams] *
                            T(Paren)[Params] * ~T(Type)[Type]) *
                 T(Group)[rhs] * End)) >>
        [](Match& _) {
          _.def(id, apply);
          return _(id = Function)
            << _(id) << (Typeparams << *_[Typeparams]) << (Params << *_[Params])
            << typevar(_, Params, Type) << (Funcbody << _[rhs]);
        },

      // (group name square parens type brace)
      In(Classbody) * T(Group)
          << (~Name[id] * ~T(Square)[Typeparams] * T(Paren)[Params] *
              ~T(Type)[Type] * ~T(Brace)[Funcbody] * (Any++)[rhs]) >>
        [](Match& _) {
          _.def(id, apply);
          return Seq << (_(id = Function)
                         << _(id) << (Typeparams << *_[Typeparams])
                         << (Params << *_[Params]) << typevar(_, Params, Type)
                         << (Funcbody << *_[Funcbody]))
                     << (Group << _[rhs]);
        },

      // Typeparams.
      T(Typeparams) << T(List)[Typeparams] >>
        [](Match& _) { return Typeparams << *_[Typeparams]; },

      // Typeparam: (group ident type)
      In(Typeparams) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = Typeparam) << _(id) << typevar(_, id, Type) << Type;
        },

      // Typeparam: (equals (group ident type) group)
      In(Typeparams) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](Match& _) {
          return _(id = Typeparam)
            << _(id) << typevar(_, id, Type) << (Type << *_[rhs]);
        },

      // Params.
      T(Params) << T(List)[Params] >>
        [](Match& _) { return Params << *_[Params]; },

      // Param: (group ident type)
      In(Params) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = Param) << _(id) << typevar(_, id, Type) << Expr;
        },

      // Param: (equals (group ident type) group)
      In(Params) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[Expr] * End) >>
        [](Match& _) {
          return _(id = Param)
            << _(id) << typevar(_, id, Type) << (Expr << *_[Expr]);
        },

      // Use.
      (In(Classbody) / In(Funcbody)) * T(Group)
          << T(Use)[Use] * (Any++)[Type] >>
        [](Match& _) { return (Use ^ _(Use)) << (Type << _[Type]); },

      // Typealias.
      (In(Classbody) / In(Funcbody)) * T(Group)
          << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * End) >>
        [](Match& _) {
          return _(id = Typealias) << _(id) << (Typeparams << *_[Typeparams])
                                   << typevar(_, id, Type) << Type;
        },

      // Typealias: (equals (typealias typeparams type type) group)
      (In(Classbody) / In(Funcbody)) * T(Equals)
          << ((T(Group)
               << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
                   ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](Match& _) {
          return _(id = Typealias) << _(id) << (Typeparams << *_[Typeparams])
                                   << typevar(_, id, Type) << (Type << *_[rhs]);
        },

      // Class.
      (In(Top) / In(Classbody) / In(Funcbody)) * T(Group)
          << (T(Class) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * T(Brace)[Classbody] * (Any++)[rhs]) >>
        [](Match& _) {
          return Seq << (_(id = Class)
                         << _(id) << (Typeparams << *_[Typeparams])
                         << (_[Type] | Type) << (Classbody << *_[Classbody]))
                     << (Group << _[rhs]);
        },

      // Type structure.
      TypeStruct * T(Group)[Type] >> [](Match& _) { return Type << *_[Type]; },
      TypeStruct * T(List)[TypeTuple] >>
        [](Match& _) { return TypeTuple << *_[TypeTuple]; },
      TypeStruct * T(Paren)[Type] >> [](Match& _) { return Type << *_[Type]; },

      // Anonymous types.
      TypeStruct * T(Brace)[Classbody] >>
        [](Match& _) { return TypeTrait << (Classbody << *_[Classbody]); },

      // Expression structure.
      ExprStruct * T(Group)[Expr] >> [](Match& _) { return Expr << *_[Expr]; },
      ExprStruct * T(List)[Tuple] >>
        [](Match& _) { return Tuple << *_[Tuple]; },
      ExprStruct * T(Equals)[Assign] >>
        [](Match& _) { return Assign << *_[Assign]; },

      // Empty parens are an empty tuple.
      ExprStruct * T(Paren) << End >> [](Match& _) -> Node { return Tuple; },
      ExprStruct* T(Paren)[Expr] >> [](Match& _) { return Expr << *_[Expr]; },

      // Typearg structure.
      (In(Type) / ExprStruct) * T(Square)[Typeargs] >>
        [](Match& _) { return Typeargs << *_[Typeargs]; },
      T(Typeargs) << T(List)[Typeargs] >>
        [](Match& _) { return Typeargs << *_[Typeargs]; },
      In(Typeargs) * T(Group)[Type] >>
        [](Match& _) { return Type << *_[Type]; },
      In(Typeargs) * T(Paren)[Type] >>
        [](Match& _) { return Type << *_[Type]; },

      // Lambda: (group typeparams) (list params...) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(List)[Params]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << *_[Params]) << (Funcbody << _[rhs]);
        },

      // Lambda: (group typeparams) (group param) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(Group)[Param]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << _[Param]) << (Funcbody << _[rhs]);
        },

      // Lambda: (list (group typeparams? param) params...) => rhs
      ExprStruct * T(Brace)
          << ((T(List)
               << ((T(Group) << (~T(Square)[Typeparams] * (Any++)[Param])) *
                   (Any++)[Params]))) *
            (T(Group) << T(FatArrow)) * (Any++)[rhs] >>
        [](Match& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (Funcbody << _[rhs]);
        },

      // Lambda: (group typeparams? param) => rhs
      ExprStruct * T(Brace)
          << ((T(Group) << (~T(Square)[Typeparams] * (Any++)[Param])) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](Match& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (Funcbody << _[rhs]);
        },

      // Zero argument lambda.
      ExprStruct * T(Brace) << (!(T(Group) << T(FatArrow)))++[Lambda] >>
        [](Match& _) {
          return Lambda << Typeparams << Params << (Funcbody << _[Lambda]);
        },

      // Var.
      ExprStruct * T(Var)[Var] * T(Ident)[id] * ~T(Type)[Type] >>
        [](Match& _) { return _(id = Var) << _(id) << typevar(_, Var, Type); },

      // Let.
      ExprStruct * T(Let)[Let] * T(Ident)[id] * ~T(Type)[Type] >>
        [](Match& _) { return _(id = Let) << _(id) << typevar(_, Let, Type); },

      // Throw.
      ExprStruct * T(Throw) * (Any++)[rhs] >>
        [](Match& _) { return Throw << (Expr << _[rhs]); },

      // Move a ref to the last expr of a sequence.
      ExprStruct * T(Ref) * T(Expr)[Expr] >>
        [](Match& _) { return Expr << Ref << *_[Expr]; },
      In(Expr) * T(Ref) * T(Expr)[lhs] * T(Expr)[rhs] >>
        [](Match& _) { return Seq << _[lhs] << Ref << _[rhs]; },
      In(Expr) * T(Ref) * T(Expr)[Expr] * End >>
        [](Match& _) { return Expr << Ref << *_[Expr]; },

      // Sequence of expr|tuple|assign in an expr.
      In(Expr) * SeqStruct[lhs] * SeqStruct[rhs] >>
        [](Match& _) {
          auto e = _(lhs);
          return Seq << (Lift << Funcbody << _(lhs)) << _[rhs];
        },

      // Remove empty groups.
      T(Group) << End >> [](Match& _) -> Node { return {}; },
    };
  }

  PassDef include()
  {
    return {
      T(Use)[lhs] << T(Type)[rhs] >>
        [](Match& _) {
          auto site = Include ^ _(lhs);
          _.include(site, look->at(_[rhs]).def);
          return site << _[rhs];
        },
    };
  }

  Token reftype(Node def)
  {
    static std::map<Token, Token> map{
      {Var, RefVar},
      {Let, RefLet},
      {Param, RefParam},
      {Class, RefType},
      {Typealias, RefType},
      {Typeparam, RefType},
      {Function, RefFunction},
    };

    if (!def)
      return Selector;

    auto it = map.find(def->type());
    if (it == map.end())
      return Selector;

    return it->second;
  }

  PassDef reftype()
  {
    return {
      dir::bottomup,
      {
        TypeStruct * (T(Ident)[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](Match& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[id] << (_[Typeargs] | Typeargs);
          },

        TypeStruct *
            (T(RefType)[lhs] * T(DoubleColon) * T(Ident)[id] *
             ~T(Typeargs)[Typeargs])[Type] >>
          [](Match& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[lhs] << _[id] << (_[Typeargs] | Typeargs);
          },
      }};
  }

  inline const auto TypeElem = T(Type) / T(RefType) / T(TypeTuple) / T(Iso) /
    T(Imm) / T(Mut) / T(TypeView) / T(TypeFunc) / T(TypeThrow) / T(TypeIsect) /
    T(TypeUnion) / T(TypeVar) / T(TypeTrait) / T(DontCare);

  PassDef typeexpr()
  {
    return {
      TypeStruct * TypeElem[lhs] * T(Symbol, "~>") * TypeElem[rhs] *
          --T(Symbol, "~>") >>
        [](Match& _) { return TypeView << _[lhs] << _[rhs]; },
      TypeStruct * TypeElem[lhs] * T(Symbol, "->") * TypeElem[rhs] *
          --T(Symbol, "->") >>
        [](Match& _) { return TypeFunc << _[lhs] << _[rhs]; },
    };
  }

  PassDef typealg()
  {
    return {
      // Build algebraic types.
      TypeStruct * TypeElem[lhs] * T(Symbol, "&") * TypeElem[rhs] >>
        [](Match& _) { return TypeIsect << _[lhs] << _[rhs]; },
      TypeStruct * TypeElem[lhs] * T(Symbol, "\\|") * TypeElem[rhs] >>
        [](Match& _) { return TypeUnion << _[lhs] << _[rhs]; },
      TypeStruct * T(Throw) * TypeElem[rhs] >>
        [](Match& _) { return TypeThrow << _[rhs]; },
      T(Type) << (TypeElem[op] * End) >> [](Match& _) { return _(op); },

      // Flatten algebraic types.
      In(TypeUnion) * T(TypeUnion)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },
      In(TypeIsect) * T(TypeIsect)[lhs] >>
        [](Match& _) { return Seq << *_[lhs]; },

    };
  }

  PassDef dnf()
  {
    return {
      T(TypeIsect)
          << (((!T(TypeUnion))++)[lhs] * T(TypeUnion)[op] * (Any++)[rhs]) >>
        [](Match& _) {
          Node r = TypeUnion;
          for (auto& child : *_(op))
            r << (TypeIsect << clone(child) << clone(_[lhs]) << clone(_[rhs]));
          return r;
        },
    };
  }

  PassDef refexpr()
  {
    return {
      dir::bottomup,
      {
        // Identifiers and symbols.
        In(Expr) * T(Dot) * Name[id] * ~T(Typeargs)[Typeargs] >>
          [](Match& _) {
            return DotSelector << _[id] << (_[Typeargs] | Typeargs);
          },

        In(Expr) * (Name[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](Match& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[id] << (_[Typeargs] | Typeargs);
          },

        // Scoped lookup.
        In(Expr) *
            (T(RefType)[lhs] * T(DoubleColon) * Name[id] *
             ~T(Typeargs)[Typeargs])[Type] >>
          [](Match& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[lhs] << _[id] << (_[Typeargs] | Typeargs);
          },

        // Create sugar.
        In(Expr) * T(RefType)[lhs] >>
          [](Match& _) {
            return Call
              << (RefFunction << _[lhs] << (Ident ^ create) << Typeargs);
          },

        // Type assertions for operators.
        // TODO: remove this?
        // In(Expr) * Start * Operator[op] * T(Type)[Type] * End >>
        //   [](Match& _) { return _(op) << _[Type]; },

        // Strip empty typeargs on variables.
        (T(RefVar) / T(RefLet) / T(RefParam))[lhs]
            << (T(Ident)[id] * (T(Typeargs) << End)) >>
          [](Match& _) { return _(lhs)->type() ^ _(id); },

        // Typeargs on variables are typeargs on apply.
        In(Expr) *
            ((T(RefVar) / T(RefLet) / T(RefParam))[lhs]
             << (T(Ident)[id] * T(Typeargs)[Typeargs])) >>
          [](Match& _) {
            return Seq << (_(lhs)->type() ^ _(id))
                       << (DotSelector << (Ident ^ apply) << _[Typeargs]);
          },

        // Compact expressions.
        T(Expr) << (T(Expr)[Expr] * End) >> [](Match& _) { return _(Expr); },
      }};
  }

  inline const auto Object = Literal / T(RefVar) / T(RefVarLHS) / T(RefLet) /
    T(RefParam) / T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) / T(Assign) /
    T(Expr) / T(DontCare);
  inline const auto Operator = T(RefFunction) / T(Selector);
  inline const auto Apply = (Selector << (Ident ^ apply) << Typeargs);
  inline const auto Load = (Selector << (Ident ^ load) << Typeargs);
  inline const auto Store = (Selector << (Ident ^ store) << Typeargs);

  inline const auto Std = (RefType << (Ident ^ standard) << Typeargs);
  inline const auto Cell = (RefType << Std << (Ident ^ cell) << Typeargs);
  inline const auto CellCreate =
    (RefFunction << Cell << (Ident ^ create) << Typeargs);
  inline const auto CallCellCreate = (Call << CellCreate);

  PassDef reverseapp()
  {
    return {
      // Dot: reverse application.
      In(Expr) * T(Tuple)[lhs] * T(DotSelector)[rhs] >>
        [](Match& _) { return Call << (Selector << *_[rhs]) << *_[lhs]; },
      In(Expr) * Object[lhs] * T(DotSelector)[rhs] >>
        [](Match& _) { return Call << (Selector << *_[rhs]) << _[lhs]; },

      In(Expr) * T(Tuple)[lhs] * T(Dot) * Operator[op] >>
        [](Match& _) { return Call << _[op] << *_[lhs]; },
      In(Expr) * Object[lhs] * T(Dot) * Operator[op] >>
        [](Match& _) { return Call << _[op] << _[lhs]; },

      In(Expr) * T(Tuple)[lhs] * T(Dot) * Object[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[rhs] << *_[lhs]; },
      In(Expr) * Object[lhs] * T(Dot) * Object[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[rhs] << _[lhs]; },
    };
  }

  PassDef application()
  {
    return {
      // Adjacency: application.
      In(Expr) * Object[lhs] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[lhs] << *_[rhs]; },
      In(Expr) * Object[lhs] * Object[rhs] >>
        [](Match& _) { return Call << clone(Apply) << _[lhs] << _[rhs]; },

      // Ref expressions.
      In(Expr) * T(Ref) * T(RefVar)[RefVar] >>
        [](Match& _) { return RefVarLHS ^ _(RefVar); },
      In(Expr) * T(Ref) * T(Call)[Call] >>
        [](Match& _) { return CallLHS << *_[Call]; },
      In(Expr) * T(Ref) * (T(Expr) << (T(Call)[Call] * End)) >>
        [](Match& _) { return CallLHS << *_[Call]; },

      // Prefix.
      In(Expr) * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << *_[rhs]; },
      In(Expr) * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << _[rhs]; },

      // Infix.
      In(Expr) * T(Tuple)[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << *_[lhs] << *_[rhs]; },
      In(Expr) * T(Tuple)[lhs] * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << *_[lhs] << _[rhs]; },
      In(Expr) * Object[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](Match& _) { return Call << _[op] << _[lhs] << *_[rhs]; },
      In(Expr) * Object[lhs] * Operator[op] * Object[rhs] >>
        [](Match& _) { return Call << _[op] << _[lhs] << _[rhs]; },
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
      (In(Funcbody) / InContainer) * T(Var)[Var] >>
        [](Match& _) {
          auto var = _(Var);
          auto id = var->at(wf / Var / Ident)->location();
          auto t = var->at(wf / Var / Type);
          return Seq << letbind(_, id, t, CallCellCreate) << (RefLet ^ id);
        },

      (In(Funcbody) / InContainer) * T(RefVar)[RefVar] >>
        [](Match& _) { return Call << clone(Load) << (RefLet ^ _(RefVar)); },

      (In(Funcbody) / InContainer) * T(RefVarLHS)[RefVarLHS] >>
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
          auto id = e->fresh();
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
                                          << Typeargs)
                                      << (RefLet ^ id))));
          }

          return Seq << letbind(_, id, typevar(_, rhs, rtype), e)
                     << (Expr << tuple << _[ltype]);
        },
    };
  }

  inline const auto LiftExpr = T(Tuple) / T(Lambda) / T(Call) / T(CallLHS);
  inline const auto RHSExpr = T(RefLet) / T(RefParam) / Literal;

  PassDef anf()
  {
    return {
      // Lift an expression as a let with a type assertion.
      (In(Funcbody) / InContainer) * LiftExpr[Lift] * ~T(Type)[Type] >>
        [](Match& _) {
          auto id = _(Lift)->fresh();
          return Seq << letbind(_, id, typevar(_, Lift, Type), _(Lift))
                     << (RefLet ^ id);
        },

      // Lift type assertions on RefLet, RefParam, and literals as new lets.
      (In(Funcbody) / InContainer) * RHSExpr[Lift] * T(Type)[Type] >>
        [](Match& _) {
          auto id = _(Lift)->fresh();
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
          return Seq << (Lift << Funcbody << (_(lhs) << _(rhs)))
                     << (RefLet ^ _(lhs)->at(wf / Let / Ident));
        },

      // Assignment to RefLet or RefParam.
      In(Assign) * (T(RefLet) / T(RefParam))[lhs] * RHSExpr[rhs] * End >>
        [](Match& _) {
          auto id = _(lhs)->fresh();
          auto t = _(lhs)->fresh();
          auto e = Call << clone(Store) << _(lhs) << _(rhs);
          return Seq << letbind(_, id, TypeVar ^ t, e) << (RefLet ^ id);
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
        {"types", types(), wfPassTypes()},
        {"structure", structure(), wfPassStructure()},
        {"include", include(), {}},
        {"reftype", reftype(), {}},
        {"typeexpr", typeexpr(), {}},
        {"typealg", typealg(), {}},
        {"dnf", dnf(), {}},
        {"refexpr", refexpr(), {}},
        {"reverseapp", reverseapp(), {}},
        {"application", application(), {}},
        {"assignlhs", assignlhs(), {}},
        {"localvar", localvar(), {}},
        {"destructure", destructure(), {}},
        {"anf", anf(), {}},
        {"assignment", assignment(), wf()},
        {"infer", infer(), {}},
      });

    return d;
  }
}
