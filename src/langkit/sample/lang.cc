#include "lang.h"

namespace verona::lang
{
  constexpr size_t restart = 0;
  const std::initializer_list<Token> terminators = {Equals, List};

  Parse parser()
  {
    Parse p("verona", Parse::depth::subdirectories);
    auto indent = std::make_shared<std::vector<std::pair<size_t, bool>>>();
    auto depth = std::make_shared<size_t>(0);

    p.preprocess([indent, depth]() {
      indent->clear();
      indent->push_back({restart, false});
      *depth = 0;
    });

    p("start",
      {
        // Blank lines terminate.
        "\n(?:[[:blank:]]*\n)+([[:blank:]]*)" >>
          [indent](auto& m) {
            indent->back() = {m.match().length(1), false};
            m.term(terminators);
          },

        // A newline that starts a brace block doesn't terminate.
        "\n([[:blank:]]*\\{[[:blank:]]*)" >>
          [indent](auto& m) {
            indent->push_back({m.match().length(1), false});
            m.pos() += m.len() - 1;
            m.len() = 1;
            m.push(Brace);
          },

        // A newline sometimes terminates.
        "\n([[:blank:]]*)" >>
          [indent](auto& m) {
            auto col = m.match().length(1);
            auto prev = indent->back().first;

            // If following a brace, don't terminate, but reset indentation.
            if (m.previous(Brace))
            {
              indent->back() = {col, false};
              return;
            }

            // Set as a continuation expression and don't terminate if:
            // * in a list
            // * in a group and indented
            if (
              m.in(List) ||
              (m.in(Group) && (col > prev) ||
               ((col == prev) && indent->back().second)))
            {
              indent->back() = {prev, true};
              return;
            }

            // Otherwise, terminate.
            indent->back() = {col, false};
            m.term(terminators);
          },

        // Whitespace between tokens.
        "[[:blank:]]+" >> [](auto& m) {},

        // Terminator.
        ";" >>
          [indent](auto& m) {
            indent->back() = {restart, false};
            m.term(terminators);
          },

        // FatArrow.
        "=>" >>
          [indent](auto& m) {
            indent->back() = {m.linecol().second + 1, false};
            m.term(terminators);
            m.add(FatArrow);
            m.term(terminators);
          },

        // Equals.
        "=" >> [](auto& m) { m.seq(Equals); },

        // List.
        "," >> [](auto& m) { m.seq(List, {Equals}); },

        // Blocks.
        "\\(([[:blank:]]*)" >>
          [indent](auto& m) {
            indent->push_back(
              {m.linecol().second + m.match().length(1), false});
            m.push(Paren);
          },

        "\\)" >>
          [indent](auto& m) {
            indent->pop_back();
            m.term(terminators);
            m.pop(Paren);
          },

        "\\[([[:blank:]]*)" >>
          [indent](auto& m) {
            indent->push_back(
              {m.linecol().second + m.match().length(1), false});
            m.push(Square);
          },

        "\\]" >>
          [indent](auto& m) {
            indent->pop_back();
            m.term(terminators);
            m.pop(Square);
          },

        "\\{([[:blank:]]*)" >>
          [indent](auto& m) {
            indent->push_back(
              {m.linecol().second + m.match().length(1), false});
            m.push(Brace);
          },

        "\\}" >>
          [indent](auto& m) {
            // A brace block terminates a fat arrow as well.
            indent->pop_back();
            m.term({Equals, List, FatArrow});
            m.pop(Brace);
          },

        // Bool.
        "(?:true|false)\\b" >> [](auto& m) { m.add(Bool); },

        // Hex float.
        "0x[[:xdigit:]]+\\.[[:xdigit:]]+(?:p[+-][[:digit:]]+)?\\b" >>
          [](auto& m) { m.add(HexFloat); },

        // Hex.
        "0x[_[:xdigit:]]+\\b" >> [](auto& m) { m.add(Hex); },

        // Bin.
        "0b[_01]+\\b" >> [](auto& m) { m.add(Bin); },

        // Float.
        "[[]:digit:]]+\\.[[:digit:]]+(?:e[+-]?[[:digit:]]+)?\\b" >>
          [](auto& m) { m.add(Float); },

        // Int.
        "[[:digit:]]+\\b" >> [](auto& m) { m.add(Int); },

        // Escaped string.
        "\"[^\"]*\"" >> [](auto& m) { m.add(Escaped); },

        // Unescaped string.
        "('+)\"[\\s\\S]*?\"\\1" >> [](auto& m) { m.add(String); },

        // Character literal.
        "'[^']*'" >> [](auto& m) { m.add(Char); },

        // Line comment.
        "//[^\n]*\n" >>
          [](auto& m) {
            // m.add(Comment);
          },

        // Nested comment.
        "/\\*" >>
          [](auto& m) {
            m.mode("comment");
            // m.add(Comment);
          },

        // Keywords.
        "private\\b" >> [](auto& m) { m.add(Private); },
        "package\\b" >> [](auto& m) { m.add(Package); },
        "using\\b" >> [](auto& m) { m.add(Using); },
        "type\\b" >> [](auto& m) { m.add(Typealias); },
        "class\\b" >> [](auto& m) { m.add(Class); },
        "var\\b" >> [](auto& m) { m.add(Var); },
        "let\\b" >> [](auto& m) { m.add(Let); },
        "throw\\b" >> [](auto& m) { m.add(Throw); },
        "iso\\b" >> [](auto& m) { m.add(Iso); },
        "imm\\b" >> [](auto& m) { m.add(Imm); },
        "mut\\b" >> [](auto& m) { m.add(Mut); },

        // Identifier.
        "[_[:alpha:]][_[:alnum:]]*\\b" >> [](auto& m) { m.add(Ident); },

        // Ellipsis.
        "\\.\\.\\." >> [](auto& m) { m.add(Ellipsis); },

        // Dot.
        "\\." >> [](auto& m) { m.add(Dot); },

        // Double colon.
        "::" >> [](auto& m) { m.add(DoubleColon); },

        // Colon.
        ":" >> [](auto& m) { m.add(Colon); },

        // Symbol. Reserved: "'(),.:;[]_{}
        "[!#$%&*+-/<=>?@\\^`|~]+" >> [](auto& m) { m.add(Symbol); },
      });

    p("comment",
      {
        "[\\s\\S]*/\\*" >> [depth](auto& m) { (*depth)++; },
        "[\\s\\S]*\\*/" >>
          [depth](auto& m) {
            if ((*depth)-- == 0)
            {
              m.extend();
              m.mode("start");
            }
          },
      });

    return p;
  }

  const auto TypeElem = T(Type) / T(TypeRef) / T(TypeTuple) / T(Iso) / T(Imm) /
    T(Mut) / T(TypeView) / T(TypeFunc) / T(TypeThrow) / T(TypeIsect) /
    T(TypeUnion);
  const auto Name = T(Ident) / T(Symbol);
  const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) / T(Hex) /
    T(Bin) / T(Int) / T(Float) / T(HexFloat);
  const auto Object = Literal / T(RefVar) / T(RefLet) / T(RefParam) / T(Tuple) /
    T(Lambda) / T(Call) / T(Expr);
  const auto Operator = T(RefFunction) / T(Selector);

  Pass imports()
  {
    return {
      // Packages.
      T(Package) * (T(String) / T(Escaped))[String] >>
        [](auto& _) { return Package << _[String]; },
    };
  }

  Pass structure()
  {
    /*
    TODO:

    = in an initializer
    dependent types

    std
      containers, iterators, network, time, strings, regex, env, ...
    builtin
      numbers, arrays, ambient authority
    public/private
    interface
    param: values as parameters for pattern matching
      named parameters
        (group ident type)
        (equals (group ident type) group*)
      pattern match on type
        (type)
      pattern match on value
        (expr)
    :: lookup in typealiases
    */
    return {
      // Module.
      T(Directory)[Directory] >>
        [](auto& _) {
          auto ident = path::last(_(Directory)->location.source->origin());
          return Class << Ident(ident) << Typeparams << Type
                       << (Classbody << *_[Directory]);
        },

      // File on its own (no module).
      In(Group) * T(File)[File] >>
        [](auto& _) {
          auto ident = path::last(_(File)->location.source->origin());
          return Class << Ident(ident) << Typeparams << Type
                       << (Classbody << *_[File]);
        },

      // File.
      T(Classbody) << (T(File) * (T(File)++)[File] * End) >>
        [](auto& _) { return Classbody << *_[File]; },

      // Type.
      T(Colon) * ((!T(Brace))++)[Type] >>
        [](auto& _) { return Type << _[Type]; },

      // Field: (group let ident type)
      In(Classbody) * T(Group) << (T(Let) * T(Ident)[id] * ~T(Type) * End) >>
        [](auto& _) {
          return _(id = FieldLet) << _[id] << (_[Type] | Type) << Expr;
        },

      // Field: (group var ident type)
      In(Classbody) * T(Group) << (T(Var) * T(Ident)[id] * ~T(Type) * End) >>
        [](auto& _) {
          return _(id = FieldVar) << _[id] << (_[Type] | Type) << Expr;
        },

      // Field: (equals (group var ident type) group)
      In(Classbody) * T(Equals)
          << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = FieldVar)
            << _[id] << (_[Type] | Type) << (Expr << *_[rhs]);
        },

      // Field: (equals (group let ident type) group)
      In(Classbody) * T(Equals)
          << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = FieldLet)
            << _[id] << (_[Type] | Type) << (Expr << *_[rhs]);
        },

      // Function.
      In(Classbody) * T(Group)
          << (~Name[id] * ~T(Square)[Typeparams] * T(Paren)[Params] *
              ~T(Type)[Type] * ~T(Brace)[Funcbody] * End) >>
        [](auto& _) {
          _.def(id, apply);
          return _(id = Function)
            << (_[id] | Ident(apply)) << (Typeparams << *_[Typeparams])
            << (Params << *_[Params]) << (_[Type] | Type)
            << (Funcbody << *_[Funcbody]);
        },

      // Typeparams.
      T(Typeparams) << T(List)[Typeparams] >>
        [](auto& _) { return Typeparams << *_[Typeparams]; },

      // Typeparam: (group ident type)
      In(Typeparams) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Typeparam) << _[id] << (_[Type] | Type) << Type;
        },

      // Typeparam: (equals (group ident type) group)
      In(Typeparams) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = Typeparam)
            << _[id] << (_[Type] | Type) << (Type << *_[rhs]);
        },

      // Params.
      T(Params) << T(List)[Params] >>
        [](auto& _) { return Params << *_[Params]; },

      // Param: (group ident type)
      In(Params) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Param) << _[id] << (_[Type] | Type) << Expr;
        },

      // Param: (equals (group ident type) group)
      In(Params) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[Expr] * End) >>
        [](auto& _) {
          return _(id = Param)
            << _[id] << (_[Type] | Type) << (Expr << *_[Expr]);
        },

      // Typealias.
      T(Group)
          << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
              T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Typealias)
            << _[id] << (Typeparams << *_[Typeparams]) << _[Type];
        },

      // Class.
      T(Group)
          << (T(Class) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * T(Brace)[Classbody] * End) >>
        [](auto& _) {
          return _(id = Class)
            << _[id] << (Typeparams << *_[Typeparams]) << (_[Type] | Type)
            << (Classbody << *_[Classbody]);
        },

      // Type.
      In(Type) * T(Group)[Type] >> [](auto& _) { return Type << *_[Type]; },
      In(Type) * T(List)[TypeTuple] >>
        [](auto& _) { return TypeTuple << *_[TypeTuple]; },
      In(Type) * T(Paren)[Type] >> [](auto& _) { return Type << *_[Type]; },

      // Typeargs.
      In(Typeargs) * T(Group)[Type] >> [](auto& _) { return Type << *_[Type]; },
      In(Typeargs) * T(List)[TypeTuple] >>
        [](auto& _) { return TypeTuple << *_[TypeTuple]; },
      In(Typeargs) * T(Paren)[Type] >> [](auto& _) { return Type << *_[Type]; },

      // Type tuple.
      In(TypeTuple) * T(Group)[Type] >>
        [](auto& _) { return Type << *_[Type]; },

      // Type scoping.
      In(Type) * T(Ident)[id] * ~T(Square)[Typeargs] >>
        [](auto& _) { return TypeRef << _[id] << (Typeargs << *_[Typeargs]); },
      In(Type) * T(TypeRef)[lhs] * T(DoubleColon) * T(Ident)[id] *
          ~T(Square)[Typeargs] >>
        [](auto& _) {
          return TypeRef << *_[lhs] << _[id] << (Typeargs << *_[Typeargs]);
        },
      In(Type) * T(Package)[Package] * T(DoubleColon) * T(Ident)[id] *
          ~T(Square)[Typeargs] >>
        [](auto& _) {
          return TypeRef << _[Package] << _[id] << (Typeargs << *_[Typeargs]);
        },

      // Type expressions.
      In(Type) * TypeElem[lhs] * R(Symbol, "~>") * TypeElem[rhs] >>
        [](auto& _) { return TypeView << _[lhs] << _[rhs]; },

      In(Type) * TypeElem[lhs] * R(Symbol, "->") * TypeElem[rhs] >>
        [](auto& _) { return TypeFunc << _[lhs] << _[rhs]; },

      In(Type) * TypeElem[lhs] * R(Symbol, "&") * TypeElem[rhs] >>
        [](auto& _) { return TypeIsect << _[lhs] << _[rhs]; },

      In(Type) * TypeElem[lhs] * R(Symbol, "\\|") * TypeElem[rhs] >>
        [](auto& _) { return TypeUnion << _[lhs] << _[rhs]; },

      In(Type) * T(Throw) * TypeElem[rhs] >>
        [](auto& _) { return TypeThrow << _[rhs]; },

      // Expression.
      In(Funcbody) * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      In(Funcbody) * T(List)[Tuple] >>
        [](auto& _) { return Tuple << *_[Tuple]; },
      In(Funcbody) * T(Equals)[Assign] >>
        [](auto& _) { return Assign << *_[Assign]; },

      In(Assign) * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      In(Assign) * T(List)[Tuple] >> [](auto& _) { return Tuple << *_[Tuple]; },

      In(Tuple) * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },

      In(Expr) * T(Paren)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      In(Expr) * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      In(Expr) * T(List)[Tuple] >> [](auto& _) { return Tuple << *_[Tuple]; },
      In(Expr) * T(Equals)[Assign] >>
        [](auto& _) { return Assign << *_[Assign]; },

      // Lambda: (group typeparams) (list params...) => rhs
      In(Expr) * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(List)[Params]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << *_[Params]) << (Funcbody << _[rhs]);
        },

      // Lambda: (group typeparams) (group param) => rhs
      In(Expr) * T(Brace)
          << (((T(Group) << T(Square)[Typeparams]) * T(Group)[Param]) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << _[Param]) << (Funcbody << _[rhs]);
        },

      // Lambda: (list (group typeparams? param) params...) => rhs
      In(Expr) * T(Brace)
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
      In(Expr) * T(Brace)
          << ((T(Group) << (~T(Square)[Typeparams] * (Any++)[Param])) *
              (T(Group) << T(FatArrow)) * (Any++)[rhs]) >>
        [](auto& _) {
          return Lambda << (Typeparams << *_[Typeparams])
                        << (Params << (Group << _[Param]) << _[Params])
                        << (Funcbody << _[rhs]);
        },

      // Zero argument lambda.
      In(Expr) * T(Brace) << (!(T(Group) << T(FatArrow)))++[Lambda] >>
        [](auto& _) {
          return Lambda << Typeparams << Params << (Funcbody << _[Lambda]);
        },

      // Var.
      In(Expr) * Start * T(Var) * T(Ident)[id] >>
        [](auto& _) { return _(id = Var) << _[id]; },

      // Let.
      In(Expr) * Start * T(Let) * T(Ident)[id] >>
        [](auto& _) { return _(id = Let) << _[id]; },

      // Throw.
      In(Expr) * Start * T(Throw) * (Any++)[rhs] >>
        [](auto& _) { return Throw << (Expr << _[rhs]); },

      // Ref.
      In(Expr) * S(Ident, Var)[id] >> [](auto& _) { return RefVar << _[id]; },
      In(Expr) * S(Ident, Let)[id] >> [](auto& _) { return RefLet << _[id]; },
      In(Expr) * S(Ident, Param)[id] >>
        [](auto& _) { return RefParam << _[id]; },
      In(Expr) * S(Ident, Typealias)[id] * ~T(Square)[Typeargs] >>
        [](auto& _) { return RefType << _[id] << (Typeargs << *_[Typeargs]); },
      In(Expr) * S(Ident, Class)[id] * ~T(Square)[Typeargs] >>
        [](auto& _) { return RefClass << _[id] << (Typeargs << *_[Typeargs]); },
      In(Expr) * (S(Ident, Function) / S(Symbol, Function))[id] *
          ~T(Square)[Typeargs] >>
        [](auto& _) {
          return RefFunction << _[id] << (Typeargs << *_[Typeargs]);
        },

      // Scoped class lookup.
      In(Expr) *
          (T(RefClass)
           << (S(Ident, Class) * T(Typeargs) *
               (L(Ident, Class) * T(Typeargs))++))[RefClass] *
          T(DoubleColon) * L(Ident, Class)[id] * ~T(Square)[Typeargs] >>
        [](auto& _) {
          return RefClass << *_[RefClass] << _[id]
                          << (Typeargs << *_[Typeargs]);
        },

      // Scoped function lookup.
      In(Expr) *
          (T(RefClass)
           << (S(Ident, Class) * T(Typeargs) *
               (L(Ident, Class) * T(Typeargs))++))[RefClass] *
          T(DoubleColon) * (L(Ident, Function) / L(Symbol, Function))[id] *
          ~T(Square)[Typeargs] >>
        [](auto& _) {
          return RefFunction << *_[RefClass] << _[id]
                             << (Typeargs << *_[Typeargs]);
        },

      // Create sugar.
      In(Expr) * T(RefClass)[RefClass] >>
        [](auto& _) {
          return Call << (RefFunction << *_[RefClass] << Ident(create)
                                      << Typeargs)
                      << Expr;
        },
    };
  }

  Pass selectors()
  {
    return {
      // Unknown names are selectors.
      In(Expr) * Name[id] * ~T(Square)[Typeargs] >>
        [](auto& _) { return Selector << _[id] << (Typeargs << *_[Typeargs]); },

      // Compact an expr that contains a single child.
      T(Expr) << (Any[Expr] * End) >> [](auto& _) { return _(Expr); },

      // Compact a type that contains a single child.
      T(Type) << (Any[Type] * End) >> [](auto& _) { return _(Type); },
    };
  }

  Pass reverseapp()
  {
    return {
      // Dot: reverse application.
      In(Expr) * Object[lhs] * T(Dot) * Any[rhs] >>
        [](auto& _) { return Call << _[rhs] << _[lhs]; },
    };
  }

  Pass expressions()
  {
    return {
      // Adjacency: application.
      In(Expr) * Object[lhs] * Object[rhs] >>
        [](auto& _) { return Call << _[lhs] << _[rhs]; },

      // Prefix.
      In(Expr) * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << _[rhs]; },

      // Infix.
      In(Expr) * T(Tuple)[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << _[op] << (Tuple << *_[lhs] << *_[rhs]); },
      In(Expr) * T(Tuple)[lhs] * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << (Tuple << *_[lhs] << _[rhs]); },
      In(Expr) * Object[lhs] * Operator[op] * T(Tuple)[rhs] >>
        [](auto& _) { return Call << _[op] << (Tuple << _[lhs] << *_[rhs]); },
      In(Expr) * Object[lhs] * Operator[op] * Object[rhs] >>
        [](auto& _) { return Call << _[op] << (Tuple << _[lhs] << _[rhs]); },
    };
  }

  Driver& driver()
  {
    static Driver d(
      "Verona",
      parser(),
      {
        {Imports, imports()},
        {Structure, structure()},
        {Selectors, selectors()},
        {ReverseApp, reverseapp()},
        {Expressions, expressions()},
      });

    return d;
  }
}
