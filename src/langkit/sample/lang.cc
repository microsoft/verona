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
        [](auto& _) {
          auto ident = path::last(_(Directory)->location().source->origin());
          return Group << (Class ^ _(Directory)) << (Ident ^ ident)
                       << (Brace << *_[File]);
        },

      // File on its own (no module). This rewrites to a class to prevent it
      // from being placed in a symbol table in the next pass.
      In(Group) * T(File)[File] >>
        [](auto& _) {
          auto ident = path::last(_(File)->location().source->origin());
          return (Class ^ ident)
            << Typeparams << Type << (Classbody << *_[File]);
        },

      // Allow disambiguating anonymous types from class or function bodies.
      In(Group) * T(Typealias) * T(Brace)[Classbody] >>
        [](auto& _) { return TypeTrait << (Classbody << *_[Classbody]); },
    };
  }

  PassDef types()
  {
    return {
      // Packages.
      T(Package) * (T(String) / T(Escaped))[String] >>
        [](auto& _) { return Package << _[String]; },

      // Type.
      T(Colon)[Colon] * ((!T(Brace))++)[Type] >>
        [](auto& _) { return (Type ^ _(Colon)) << _[Type]; },
    };
  }

  inline const auto ExprStruct =
    In(Funcbody) / In(Assign) / In(Tuple) / In(Expr) / In(Term);
  inline const auto TermStruct = In(Term) / In(Expr);
  inline const auto TypeStruct = In(Type) / In(TypeTerm) / In(TypeTuple);
  inline const auto SeqStruct = T(Expr) / T(Tuple) / T(Assign);
  inline const auto Name = T(Ident) / T(Symbol);
  inline const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) /
    T(Hex) / T(Bin) / T(Int) / T(Float) / T(HexFloat);

  auto typevar(auto& _, const Token& e, const Token& t)
  {
    return _(t) ? _(t) : TypeVar ^ _(e)->fresh();
  }

  PassDef structure()
  {
    return {
      // Let Field:
      // (equals (group let ident type) group)
      // (group let ident type)
      In(Classbody) *
          (T(Equals)
           << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
               T(Group)[rhs] * End)) /
          (T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) >>
        [](auto& _) {
          return _(id = FieldLet) << typevar(_, id, Type) << (Expr << *_[rhs]);
        },

      // Var Field:
      // (equals (group var ident type) group)
      // (group var ident type)
      In(Classbody) *
          (T(Equals)
           << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
               T(Group)[rhs] * End)) /
          (T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) >>
        [](auto& _) {
          return _(id = FieldVar) << typevar(_, id, Type) << (Expr << *_[rhs]);
        },

      // Function.
      // (equals (group name square parens type) group)
      In(Classbody) *
          (T(Equals)
           << (T(Group) << (~Name[id] * ~T(Square)[Typeparams] *
                            T(Paren)[Params] * ~T(Type)[Type]) *
                 T(Group)[rhs])) >>
        [](auto& _) {
          _.def(id, apply);
          return _(id = Function)
            << (Typeparams << *_[Typeparams]) << (Params << *_[Params])
            << typevar(_, Params, Type) << (Funcbody << _[rhs]);
        },

      // (group name square parens type brace)
      In(Classbody) * T(Group)
          << (~Name[id] * ~T(Square)[Typeparams] * T(Paren)[Params] *
              ~T(Type)[Type] * ~T(Brace)[Funcbody] * (Any++)[rhs]) >>
        [](auto& _) {
          _.def(id, apply);
          return Seq << (_(id = Function)
                         << (Typeparams << *_[Typeparams])
                         << (Params << *_[Params]) << typevar(_, Params, Type)
                         << (Funcbody << *_[Funcbody]))
                     << (Group << _[rhs]);
        },

      // Typeparams.
      T(Typeparams) << T(List)[Typeparams] >>
        [](auto& _) { return Typeparams << *_[Typeparams]; },

      // Typeparam: (group ident type)
      In(Typeparams) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Typeparam) << typevar(_, id, Type) << Type;
        },

      // Typeparam: (equals (group ident type) group)
      In(Typeparams) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = Typeparam) << typevar(_, id, Type) << (Type << *_[rhs]);
        },

      // Params.
      T(Params) << T(List)[Params] >>
        [](auto& _) { return Params << *_[Params]; },

      // Param: (group ident type)
      In(Params) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(id = Param) << typevar(_, id, Type) << Expr; },

      // Param: (equals (group ident type) group)
      In(Params) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[Expr] * End) >>
        [](auto& _) {
          return _(id = Param) << typevar(_, id, Type) << (Expr << *_[Expr]);
        },

      // Use.
      (In(Classbody) / In(Funcbody)) * T(Group)
          << T(Use)[Use] * (Any++)[Type] >>
        [](auto& _) { return (Use ^ _(Use)) << (Type << _[Type]); },

      // Typealias.
      (In(Classbody) / In(Funcbody)) * T(Group)
          << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Typealias)
            << (Typeparams << *_[Typeparams]) << typevar(_, id, Type) << Type;
        },

      // Typealias: (equals (typealias typeparams type type) group)
      (In(Classbody) / In(Funcbody)) * T(Equals)
          << ((T(Group)
               << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
                   ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = Typealias) << (Typeparams << *_[Typeparams])
                                   << typevar(_, id, Type) << (Type << *_[rhs]);
        },

      // Class.
      (In(Classbody) / In(Funcbody)) * T(Group)
          << (T(Class) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * T(Brace)[Classbody] * (Any++)[rhs]) >>
        [](auto& _) {
          return Seq << (_(id = Class)
                         << (Typeparams << *_[Typeparams]) << (_[Type] | Type)
                         << (Classbody << *_[Classbody]))
                     << (Group << _[rhs]);
        },

      // Type structure.
      TypeStruct * T(Group)[TypeTerm] >>
        [](auto& _) { return TypeTerm << *_[TypeTerm]; },
      TypeStruct * T(List)[TypeTuple] >>
        [](auto& _) { return TypeTuple << *_[TypeTuple]; },
      TypeStruct * T(Paren)[TypeTerm] >>
        [](auto& _) { return TypeTerm << *_[TypeTerm]; },

      // Anonymous types.
      TypeStruct * T(Brace)[Classbody] >>
        [](auto& _) { return TypeTrait << (Classbody << *_[Classbody]); },

      // Expression structure.
      ExprStruct * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      ExprStruct * T(List)[Tuple] >> [](auto& _) { return Tuple << *_[Tuple]; },
      ExprStruct * T(Equals)[Assign] >>
        [](auto& _) { return Assign << *_[Assign]; },
      ExprStruct * T(Paren)[Term] >> [](auto& _) { return Term << *_[Term]; },

      // An empty term is an empty tuple.
      T(Term) << End >> [](auto& _) -> Node { return Tuple; },

      // Typearg structure.
      (In(Type) / ExprStruct) * T(Square)[Typeargs] >>
        [](auto& _) { return Typeargs << *_[Typeargs]; },
      T(Typeargs) << T(List)[Typeargs] >>
        [](auto& _) { return Typeargs << *_[Typeargs]; },
      In(Typeargs) * T(Group)[Type] >> [](auto& _) { return Type << *_[Type]; },
      In(Typeargs) * T(Paren)[Type] >> [](auto& _) { return Type << *_[Type]; },

      // Lambda: (group typeparams) (list params...) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(List)[Params]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << *_[Params]) << (Funcbody << _[rhs]);
        },

      // Lambda: (group typeparams) (group param) => rhs
      ExprStruct * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(Group)[Param]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << _[Param]) << (Funcbody << _[rhs]);
        },

      // Lambda: (list (group typeparams? param) params...) => rhs
      ExprStruct * T(Brace)
          << ((T(List)
               << ((T(Group) << (~T(Square)[Typeparams] * (Any++)[Param])) *
                   (Any++)[Params]))) *
            (T(Group) << T(FatArrow)) * (Any++)[rhs] >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (Funcbody << _[rhs]);
        },

      // Lambda: (group typeparams? param) => rhs
      ExprStruct * T(Brace)
          << ((T(Group) << (~T(Square)[Typeparams] * (Any++)[Param])) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (Funcbody << _[rhs]);
        },

      // Zero argument lambda.
      ExprStruct * T(Brace) << (!(T(Group) << T(FatArrow)))++[Lambda] >>
        [](auto& _) {
          return Lambda << Typeparams << Params << (Funcbody << _[Lambda]);
        },

      // Var.
      ExprStruct * Start * T(Var) * T(Ident)[id] >>
        [](auto& _) { return _(id = Var); },

      // Let.
      ExprStruct * Start * T(Let) * T(Ident)[id] >>
        [](auto& _) { return _(id = Let); },

      // Throw.
      ExprStruct * Start * T(Throw) * (Any++)[rhs] >>
        [](auto& _) { return Throw << (Expr << _[rhs]); },

      // Move a ref to the last expr of a sequence.
      ExprStruct * T(Ref) * T(Term)[Term] >>
        [](auto& _) { return Term << Ref << *_[Term]; },
      In(Term) * T(Ref) * T(Expr)[lhs] * T(Expr)[rhs] >>
        [](auto& _) { return Seq << _[lhs] << Ref << _[rhs]; },
      In(Term) * T(Ref) * T(Expr)[Expr] * End >>
        [](auto& _) { return Expr << Ref << *_[Expr]; },

      // Sequence of expr|tuple|assign in a term.
      In(Term) * SeqStruct[lhs] * SeqStruct[rhs] >>
        [](auto& _) {
          auto e = _(lhs);
          return Seq << (Lift << Funcbody << _(lhs)) << _[rhs];
        },

      // Compact single element terms.
      T(Term) << (Any[op] * End) >> [](auto& _) { return _(op); },

      // Remove empty groups.
      T(Group) << End >> [](auto& _) -> Node { return {}; },
    };
  }

  PassDef include()
  {
    return {
      T(Use)[lhs] << T(Type)[rhs] >>
        [](auto& _) {
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
      {Class, RefClass},
      {Typealias, RefTypealias},
      {Typeparam, RefTypeparam},
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
          [](auto& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[id] << (_[Typeargs] | Typeargs);
          },

        TypeStruct *
            ((T(RefClass) / T(RefTypealias) / T(RefTypeparam) /
              T(Package))[lhs] *
             T(DoubleColon) * T(Ident)[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](auto& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[lhs] << _[id] << (_[Typeargs] | Typeargs);
          },
      }};
  }

  inline const auto TypeElem = T(TypeTerm) / T(RefClass) / T(RefTypealias) /
    T(RefTypeparam) / T(TypeTuple) / T(Iso) / T(Imm) / T(Mut) / T(TypeView) /
    T(TypeFunc) / T(TypeThrow) / T(TypeIsect) / T(TypeUnion) / T(TypeVar) /
    T(TypeTrait) / T(DontCare);

  PassDef typeexpr()
  {
    return {
      TypeStruct * TypeElem[lhs] * T(Symbol, "~>") * TypeElem[rhs] *
          --T(Symbol, "~>") >>
        [](auto& _) { return TypeView << _[lhs] << _[rhs]; },
      TypeStruct * TypeElem[lhs] * T(Symbol, "->") * TypeElem[rhs] *
          --T(Symbol, "->") >>
        [](auto& _) { return TypeFunc << _[lhs] << _[rhs]; },
    };
  }

  PassDef typealg()
  {
    return {
      TypeStruct * TypeElem[lhs] * T(Symbol, "&") * TypeElem[rhs] >>
        [](auto& _) { return TypeIsect << _[lhs] << _[rhs]; },
      TypeStruct * TypeElem[lhs] * T(Symbol, "\\|") * TypeElem[rhs] >>
        [](auto& _) { return TypeUnion << _[lhs] << _[rhs]; },
      TypeStruct * T(Throw) * TypeElem[rhs] >>
        [](auto& _) { return TypeThrow << _[rhs]; },
      T(TypeTerm) << (TypeElem[op] * End) >> [](auto& _) { return _(op); },
    };
  }

  inline const auto Operator = T(RefFunction) / T(Selector);

  PassDef refexpr()
  {
    return {
      dir::bottomup,
      {
        // Flatten algebraic types.
        In(TypeUnion) * T(TypeUnion)[lhs] >>
          [](auto& _) { return Seq << *_[lhs]; },
        In(TypeIsect) * T(TypeIsect)[lhs] >>
          [](auto& _) { return Seq << *_[lhs]; },

        // Identifiers and symbols.
        TermStruct * T(Dot) * Name[id] * ~T(Typeargs)[Typeargs] >>
          [](auto& _) {
            return DotSelector << _[id] << (_[Typeargs] | Typeargs);
          },

        TermStruct * (Name[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](auto& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[id] << (_[Typeargs] | Typeargs);
          },

        // Scoped lookup.
        TermStruct *
            ((T(RefClass) / T(RefTypealias) / T(RefTypeparam) /
              T(Package))[lhs] *
             T(DoubleColon) * Name[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](auto& _) {
            auto def = look->at(_[Type]).def;
            return reftype(def) << _[lhs] << _[id] << (_[Typeargs] | Typeargs);
          },

        // Create sugar.
        TermStruct * (T(RefClass) / T(RefTypeparam))[lhs] >>
          [](auto& _) {
            return Call
              << (RefFunction << _[lhs] << (Ident ^ create) << Typeargs);
          },

        // Type assertions for operators.
        TermStruct * Start * Operator[op] * T(Type)[Type] * End >>
          [](auto& _) { return _(op) << _[Type]; },

        // Strip empty typeargs on variables.
        (T(RefVar) / T(RefLet) / T(RefParam))[lhs]
            << (T(Ident)[id] * (T(Typeargs) << End)) >>
          [](auto& _) { return _(lhs)->type() ^ _(id); },

        // Typeargs on variables are typeargs on apply.
        TermStruct *
            ((T(RefVar) / T(RefLet) / T(RefParam))[lhs]
             << (T(Ident)[id] * T(Typeargs)[Typeargs])) >>
          [](auto& _) {
            return Seq << (_(lhs)->type() ^ _(id))
                       << (DotSelector << (Ident ^ apply) << _[Typeargs]);
          },

        // Compact expressions.
        T(Expr) << (T(Expr)[Expr] * End) >> [](auto& _) { return _(Expr); },
      }};
  }

  inline const auto Object = Literal / T(RefVar) / T(RefVarLHS) / T(RefLet) /
    T(RefParam) / T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) / T(Assign) /
    T(Term) / T(DontCare);
  inline const auto Apply = (Selector << (Ident ^ apply) << Typeargs);

  PassDef reverseapp()
  {
    return {
      // Dot: reverse application.
      TermStruct * T(Tuple)[lhs] * T(DotSelector)[rhs] >>
        [](auto& _) { return Call << (Selector << *_[rhs]) << *_[lhs]; },
      TermStruct * Object[lhs] * T(DotSelector)[rhs] >>
        [](auto& _) { return Call << (Selector << *_[rhs]) << _[lhs]; },

      TermStruct * T(Tuple)[lhs] * T(Dot) * Operator[op] >>
        [](auto& _) { return Call << _[op] << *_[lhs]; },
      TermStruct * Object[lhs] * T(Dot) * Operator[op] >>
        [](auto& _) { return Call << _[op] << _[lhs]; },

      TermStruct * T(Tuple)[lhs] * T(Dot) * Object[rhs] >>
        [](auto& _) { return Call << clone(Apply) << _[rhs] << *_[lhs]; },
      TermStruct * Object[lhs] * T(Dot) * Object[rhs] >>
        [](auto& _) { return Call << clone(Apply) << _[rhs] << _[lhs]; },
    };
  }

  PassDef application()
  {
    return {
      // Adjacency: application.
      TermStruct * Object[lhs] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << clone(Apply) << _[lhs] << *_[rhs]; },
      TermStruct * Object[lhs] * Object[rhs] >>
        [](auto& _) { return Call << clone(Apply) << _[lhs] << _[rhs]; },

      // Ref expressions.
      TermStruct * T(Ref) * T(RefVar)[RefVar] >>
        [](auto& _) { return (RefVarLHS ^ _(RefVar)) << *_[RefVar]; },
      TermStruct * T(Ref) * T(Call)[Call] >>
        [](auto& _) { return CallLHS << *_[Call]; },
      TermStruct * T(Ref) * (T(Expr) << (T(Call)[Call] * End)) >>
        [](auto& _) { return CallLHS << *_[Call]; },

      // Prefix.
      TermStruct * Operator[op] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << _[op] << *_[rhs]; },
      TermStruct * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << _[rhs]; },

      // Infix.
      TermStruct * T(Tuple)[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << _[op] << *_[lhs] << *_[rhs]; },
      TermStruct * T(Tuple)[lhs] * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << *_[lhs] << _[rhs]; },
      TermStruct * Object[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << _[op] << _[lhs] << *_[rhs]; },
      TermStruct * Object[lhs] * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << _[lhs] << _[rhs]; },
    };
  }

  inline const auto InContainer = In(Expr) / In(Term) / In(Tuple) / In(Call);

  PassDef vardecl()
  {
    return {
      // Don't leave a reference to a var or let if it's declared at the top.
      In(Funcbody) * T(Expr)
          << ((T(Var) / T(Let))[Var] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(Var) << typevar(_, Var, Type); },

      // Lift a var declaration with it's type assertion.
      InContainer * T(Var)[Var] * ~T(Type)[Type] >>
        [](auto& _) {
          return Seq << (Lift << Funcbody << (_(Var) << typevar(_, Var, Type)))
                     << (RefVar ^ _(Var));
        },
    };
  }

  inline const auto LHSExpr = T(RefLet) / T(RefVarLHS) / T(CallLHS);
  inline const auto LiftExpr =
    T(Tuple) / T(Lambda) / T(Call) / T(CallLHS) / T(Assign);
  inline const auto RefExpr =
    T(RefVar) / T(RefVarLHS) / T(RefLet) / T(RefParam) / Literal;

  PassDef assignment()
  {
    return {
      // Turn a Tuple on the LHS of an assignment into a TupleLHS.
      In(Assign) * (T(Expr) << (T(Tuple)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](auto& _) {
          return Seq << (Expr << (TupleLHS << *_[lhs]) << _[Type]) << _[rhs];
        },

      // Turn a Call on the LHS of an assignment into a CallLHS.
      In(Assign) * (T(Expr) << (T(Call)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](auto& _) {
          return Seq << (Expr << (CallLHS << *_[lhs]) << _[Type]) << _[rhs];
        },

      // Turn a RefVar on the LHS of an assignment into a RefVarLHS.
      In(Assign) * (T(Expr) << (T(RefVar)[lhs] * ~T(Type)[Type])) * Any[rhs] >>
        [](auto& _) {
          return Seq << (Expr << (RefVarLHS ^ _(lhs)) << _[Type]) << _[rhs];
        },

      // Recurse LHS.
      In(TupleLHS) * (T(Expr) << (T(Tuple)[lhs] * ~T(Type)[Type])) >>
        [](auto& _) { return Expr << (TupleLHS << *_[lhs]) << _[Type]; },

      In(TupleLHS) * (T(Expr) << (T(Call)[lhs] * ~T(Type)[Type])) >>
        [](auto& _) { return Expr << (CallLHS << *_[lhs]) << _[Type]; },

      In(TupleLHS) * (T(Expr) << (T(RefVar)[lhs] * ~T(Type)[Type])) >>
        [](auto& _) { return Expr << (RefVarLHS ^ _(lhs)) << _[Type]; },

      // Destructuring assignment.
      In(Assign) * (T(Expr) << (T(TupleLHS)[lhs] * ~T(Type)[ltype] * End)) *
          (T(Expr) << (Any[rhs] * ~T(Type)[rtype] * End)) * End >>
        [](auto& _) {
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

          return Seq << (Lift << Funcbody
                              << (_(id = Let) << typevar(_, rhs, rtype) << e))
                     << tuple << _[ltype];
        },

      // Let binding.
      In(Assign) * (T(Expr) << (T(Let)[lhs] * ~T(Type)[ltype] * End)) *
          T(Expr)[rhs] * End >>
        [](auto& _) {
          return Seq << (Lift << Funcbody
                              << (_(lhs) << typevar(_, lhs, ltype) << _(rhs)))
                     << (RefLet ^ _(lhs));
        },

      // Assignment.
      In(Assign) * (T(Expr) << (LHSExpr[lhs] * ~T(Type)[ltype] * End)) *
          (T(Expr) << (LiftExpr[rhs] * ~T(Type)[rtype] * End)) * End >>
        [](auto& _) {
          auto e0 = _(rhs);
          auto id0 = e0->fresh();
          auto t0 = typevar(_, rhs, rtype);

          auto e1 = _(lhs);
          auto id1 = e1->fresh();
          auto t1 = TypeVar ^ e1->fresh();

          auto e2 = Load ^ id1;
          auto id2 = e1->fresh();
          auto t2 = typevar(_, lhs, ltype);

          return Seq << (Lift << Funcbody << (_(id0 = Let) << t0 << e0))
                     << (Lift << Funcbody << (_(id1 = Let) << t1 << e1))
                     << (Lift << Funcbody << (_(id2 = Let) << t2 << e2))
                     << (Lift << Funcbody
                              << (Store << (RefLet ^ id1) << (RefLet ^ id0)))
                     << (Expr << (RefLet ^ id2));
        },

      In(Assign) * (T(Expr) << (LHSExpr[lhs] * ~T(Type)[ltype] * End)) *
          (T(Expr) << (RefExpr[rhs] * ~T(Type)[rtype] * End)) * End >>
        [](auto& _) {
          // TODO: what do we do with rtype?
          auto e1 = _(lhs);
          auto id1 = e1->fresh();
          auto t1 = TypeVar ^ e1->fresh();

          auto e2 = Load ^ id1;
          auto id2 = e1->fresh();
          auto t2 = typevar(_, lhs, ltype);

          return Seq << (Lift << Funcbody << (_(id1 = Let) << t1 << e1))
                     << (Lift << Funcbody << (_(id2 = Let) << t2 << e2))
                     << (Lift << Funcbody
                              << (Store << (RefLet ^ id1) << _(rhs)))
                     << (Expr << (RefLet ^ id2));
        },

      // Compact assigns after they're reduced.
      T(Assign) << (Any[op] * End) >> [](auto& _) { return _(op); },
    };
  }

  PassDef anf()
  {
    return {
      // Lift an expression as a let with a type assertion.
      (In(Funcbody) / InContainer) * LiftExpr[Lift] * ~T(Type)[Type] >>
        [](auto& _) {
          auto id = _(Lift)->fresh();
          return Seq << (Lift
                         << Funcbody
                         << (_(id = Let) << typevar(_, Lift, Type) << _(Lift)))
                     << (RefLet ^ id);
        },

      // Compact terms and exprs after they're reduced.
      (T(Term) / T(Expr)) << (Any[op] * End) >> [](auto& _) { return _(op); },
    };
  }

  PassDef dnf()
  {
    return {
      T(TypeIsect)
          << (((!T(TypeUnion))++)[lhs] * T(TypeUnion)[op] * (Any++)[rhs]) >>
        [](auto& _) {
          Node r = TypeUnion;
          for (auto& child : *_(op))
            r << (TypeIsect << clone(child) << clone(_[lhs]) << clone(_[rhs]));
          return r;
        },
    };
  }

  Driver& driver()
  {
    static Driver d(
      "Verona",
      parser(),
      {
        {"modules", modules()},
        {"types", types()},
        {"structure", structure()},
        {"include", include()},
        {"reftype", reftype()},
        {"typeexpr", typeexpr()},
        {"typealg", typealg()},
        {"refexpr", refexpr()},
        {"reverseapp", reverseapp()},
        {"application", application()},
        {"vardecl", vardecl()},
        {"assignment", assignment()},
        {"anf", anf()},
        {"dnf", dnf()},
        {"infer", infer()},
      });

    return d;
  }
}
