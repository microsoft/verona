#include "lang.h"

namespace verona::lang
{
  constexpr size_t restart = 0;
  const std::initializer_list<Token> terminators = {Equals, List};

  Parse parser()
  {
    Parse p(depth::subdirectories);
    auto depth = std::make_shared<size_t>(0);
    auto indent = std::make_shared<std::vector<std::pair<size_t, bool>>>();
    indent->push_back({restart, false});

    p.prefile(
      [](auto& p, auto& path) { return path::extension(path) == "verona"; });

    p.predir([](auto& p, auto& path) {
      static auto re = std::regex(
        ".*/[_[:alpha:]][_[:alnum:]]*/$", std::regex_constants::optimize);
      return std::regex_match(path, re);
    });

    p.postparse([](auto& p, auto ast) {
      auto stdlib = path::directory(path::executable()) + "std/";
      if (ast->location().source->origin() != stdlib)
        ast->push_back(p.parse(stdlib));
    });

    p.postfile([indent, depth](auto& p, auto ast) {
      *depth = 0;
      indent->clear();
      indent->push_back({restart, false});
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
        "\n([[:blank:]]*(\\{[[:blank:]]*))" >>
          [indent](auto& m) {
            indent->push_back({m.match().length(1), false});
            m.pos() += m.len() - m.match().length(2);
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
        "\"(?:\\\"|[^\"])*\"" >> [](auto& m) { m.add(Escaped); },

        // Unescaped string.
        "('+)\"[\\s\\S]*?\"\\1" >> [](auto& m) { m.add(String); },

        // Character literal.
        "'[^']*'" >> [](auto& m) { m.add(Char); },

        // Line comment.
        "//[^\n]*" >> [](auto& m) { m.add(Comment); },

        // Nested comment.
        "/\\*" >>
          [](auto& m) {
            m.mode("comment");
            m.add(Comment);
          },

        // Keywords.
        "private\\b" >> [](auto& m) { m.add(Private); },
        "package\\b" >> [](auto& m) { m.add(Package); },
        "use\\b" >> [](auto& m) { m.add(Use); },
        "type\\b" >> [](auto& m) { m.add(Typealias); },
        "class\\b" >> [](auto& m) { m.add(Class); },
        "var\\b" >> [](auto& m) { m.add(Var); },
        "let\\b" >> [](auto& m) { m.add(Let); },
        "throw\\b" >> [](auto& m) { m.add(Throw); },
        "iso\\b" >> [](auto& m) { m.add(Iso); },
        "imm\\b" >> [](auto& m) { m.add(Imm); },
        "mut\\b" >> [](auto& m) { m.add(Mut); },

        // Don't care.
        "_(?![_[:alnum:]])" >> [](auto& m) { m.add(DontCare); },

        // Reserve a sequence of underscores.
        "_(?:_)+(?![[:alnum:]])" >> [](auto& m) { m.add(Invalid); },

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

  const auto TypeElem = T(Type) / T(RefClass) / T(RefType) / T(RefTypeparam) /
    T(TypeTuple) / T(Iso) / T(Imm) / T(Mut) / T(TypeView) / T(TypeFunc) /
    T(TypeThrow) / T(TypeIsect) / T(TypeUnion) / T(DontCare);
  const auto Name = T(Ident) / T(Symbol);
  const auto Literal = T(String) / T(Escaped) / T(Char) / T(Bool) / T(Hex) /
    T(Bin) / T(Int) / T(Float) / T(HexFloat);
  const auto Object = Literal / T(RefVar) / T(RefLet) / T(RefParam) / T(Tuple) /
    T(Lambda) / T(Call) / T(Oftype) / T(Expr) / T(DontCare);
  const auto Operator = T(RefFunction) / T(Selector);
  const auto InExpr =
    In(Funcbody) / In(Assign) / In(Tuple) / In(Expr) / In(Call);
  const auto TypeOrExpr = In(Type) / InExpr;

  inline constexpr auto wf = wellformed(
    shape(Package, field(id, String, Escaped)),
    shape(Class, field(Typeparams), field(Type), field(Classbody)),
    shape(
      Typealias, field(Typeparams), field(Bounds, Type), field(Default, Type)),
    shape(Typeparam, field(Bounds, Type), field(Default, Type)),
    shape(
      Function, field(Typeparams), field(Params), field(Type), field(Funcbody)),
    shape(FieldLet, field(Type), field(Expr)),
    shape(FieldVar, field(Type), field(Expr)));

  Token reftype(Node def)
  {
    static std::map<Token, Token> map{
      {Var, RefVar},
      {Let, RefLet},
      {Param, RefParam},
      {Class, RefClass},
      {Typealias, RefType},
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

  Lookup lookup()
  {
    Lookup look;

    auto subs = std::make_shared<std::map<Node, Node, std::owner_less<>>>();
    look.post([subs](auto& _) { subs->clear(); });

    auto typeargs = [subs](auto& _, Node def) {
      // TODO: what if def is a Typeparam?
      // use the bounds somehow
      auto ta = _(Typeargs);
      if (!def || !ta)
        return;

      constexpr std::array<Token, 3> list{Typealias, Class, Function};
      auto it = std::find(list.begin(), list.end(), def->type());
      if (it == list.end())
        return;

      auto tp = def->at(
        wf / Typealias / Typeparams,
        wf / Class / Typeparams,
        wf / Function / Typeparams);

      std::vector<Node> args;
      std::transform(
        ta->begin(), ta->end(), std::back_inserter(args), [&_](auto& arg) {
          return _.find(arg);
        });
      args.resize(tp->size());

      std::transform(
        tp->begin(),
        tp->end(),
        args.begin(),
        std::inserter(*subs, subs->end()),
        [](auto& param, auto& arg) { return std::make_pair(param, arg); });
    };

    auto sub = [subs](Node def) {
      if (!def || (def->type() != Typeparam))
        return def;

      auto it = subs->find(def);
      if ((it != subs->end()) && it->second)
        return it->second;

      return def;
    };

    return look({
      T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [typeargs](auto& _) {
          auto def = _(id)->lookup_first();
          typeargs(_, def);
          return _.find(def);
        },

      (T(Var) / T(Let) / T(Param) / T(Class) / T(Function))[id] >>
        [](auto& _) { return _(id); },

      T(Type) << Any[Type] >> [](auto& _) { return _.find(Type); },

      T(Typealias)[id] >>
        [](auto& _) { return _.find(_(id)->at(wf / Typealias / Default)); },

      T(Typeparam)[id] >>
        [sub](auto& _) {
          auto def = sub(_(id));
          if (def->type() != Typeparam)
            return def;
          auto bounds = def->at(wf / Typeparam / Bounds);
          return bounds->empty() ? def : _.find(bounds);
        },

      (T(RefClass) / T(RefType) / T(RefTypeparam) / T(Package))[lhs] *
          T(DoubleColon) * T(Ident)[id] * ~T(Typeargs)[Typeargs] >>
        [typeargs](auto& _) {
          auto def = _.find(lhs)->lookdown_first(_(id));
          typeargs(_, def);
          return _.find(def);
        },

      (T(RefClass) / T(RefType) / T(RefTypeparam) / T(Package))
          << (T(Ident) * T(Typeargs))[id] >>
        [](auto& _) { return _.find(id); },

      (T(RefClass) / T(RefType) / T(RefTypeparam) / T(Package))
          << (Any[lhs] * T(Ident)[id] * T(Typeargs)[Typeargs]) >>
        [typeargs](auto& _) {
          auto def = _.find(lhs)->lookdown_first(_(id));
          typeargs(_, def);
          return _.find(def);
        },
    });
  }

  const auto look = lookup();

  Pass modules()
  {
    /*
    TODO:

    list inside Typeparams or Typeargs along with groups or other lists
    = in an initializer
    type compaction?
    lookup
      isect: lookup in lhs and rhs
    well-formedness for errors
    error on too many typeargs
    recursive typealias?

    DNF algebraic types
    right associative function and viewpoint types
    ANF
    type checker

    _ for partial application and higher order bounds
      T: C1[C2, _, C3] is a type parameter that takes one argument
      given f(x: A, y: B, z: C), f(_, _, z) is a function that takes (A, B)

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

    package schemes
    dependent types
    */
    return {
      // Module.
      T(Directory)[Directory] << (T(File)++)[File] >>
        [](auto& _) {
          auto ident = path::last(_(Directory)->location().source->origin());
          return Group << (Class ^ _(Directory)) << Ident(ident)
                       << (Brace << *_[File]);
        },

      // File on its own (no module). This rewrites to a class to prevent it
      // from being placed in a symbol table in the next pass.
      In(Group) * T(File)[File] >>
        [](auto& _) {
          auto ident = path::last(_(File)->location().source->origin());
          return Class(ident) << Typeparams << Type << (Classbody << *_[File]);
        },

      // Comments and empty groups.
      T(Comment) >> [](auto& _) -> Node { return {}; },
      T(Group) << End >> [](auto& _) -> Node { return {}; },
    };
  }

  Pass types()
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

  Pass structure()
  {
    return {
      // Field: (group let ident type)
      In(Classbody) * T(Group)
          << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(id = FieldLet) << (_[Type] | Type) << Expr; },

      // Field: (group var ident type)
      In(Classbody) * T(Group)
          << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(id = FieldVar) << (_[Type] | Type) << Expr; },

      // Field: (equals (group var ident type) group)
      In(Classbody) * T(Equals)
          << ((T(Group) << (T(Var) * T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = FieldVar) << (_[Type] | Type) << (Expr << *_[rhs]);
        },

      // Field: (equals (group let ident type) group)
      In(Classbody) * T(Equals)
          << ((T(Group) << (T(Let) * T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = FieldLet) << (_[Type] | Type) << (Expr << *_[rhs]);
        },

      // Function.
      In(Classbody) * T(Group)
          << (~Name[id] * ~T(Square)[Typeparams] * T(Paren)[Params] *
              ~T(Type)[Type] * ~T(Brace)[Funcbody] * End) >>
        [](auto& _) {
          _.def(id, apply);
          return _(id = Function)
            << (Typeparams << *_[Typeparams]) << (Params << *_[Params])
            << (_[Type] | Type) << (Funcbody << *_[Funcbody]);
        },

      // Typeparams.
      T(Typeparams) << T(List)[Typeparams] >>
        [](auto& _) { return Typeparams << *_[Typeparams]; },

      // Typeparam: (group ident type)
      In(Typeparams) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(id = Typeparam) << (_[Type] | Type) << Type; },

      // Typeparam: (equals (group ident type) group)
      In(Typeparams) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = Typeparam) << (_[Type] | Type) << (Type << *_[rhs]);
        },

      // Params.
      T(Params) << T(List)[Params] >>
        [](auto& _) { return Params << *_[Params]; },

      // Param: (group ident type)
      In(Params) * T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End) >>
        [](auto& _) { return _(id = Param) << (_[Type] | Type) << Expr; },

      // Param: (equals (group ident type) group)
      In(Params) * T(Equals)
          << ((T(Group) << (T(Ident)[id] * ~T(Type)[Type] * End)) *
              T(Group)[Expr] * End) >>
        [](auto& _) {
          return _(id = Param) << (_[Type] | Type) << (Expr << *_[Expr]);
        },

      // Use.
      T(Group) << T(Use)[Use] * (Any++)[Type] >>
        [](auto& _) { return (Use ^ _(Use)) << (Type << _[Type]); },

      // Typealias.
      T(Group)
          << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * End) >>
        [](auto& _) {
          return _(id = Typealias)
            << (Typeparams << *_[Typeparams]) << (_[Type] | Type) << Type;
        },

      // Typealias: (equals (group typealias ident typeparams type) group)
      T(Equals)
          << ((T(Group)
               << (T(Typealias) * T(Ident)[id] * ~T(Square)[Typeparams] *
                   ~T(Type)[Type] * End)) *
              T(Group)[rhs] * End) >>
        [](auto& _) {
          return _(id = Typealias) << (Typeparams << *_[Typeparams])
                                   << (_[Type] | Type) << (Type << *_[rhs]);
        },

      // Class.
      T(Group)
          << (T(Class) * T(Ident)[id] * ~T(Square)[Typeparams] *
              ~T(Type)[Type] * T(Brace)[Classbody] * End) >>
        [](auto& _) {
          return _(id = Class)
            << (Typeparams << *_[Typeparams]) << (_[Type] | Type)
            << (Classbody << *_[Classbody]);
        },

      // Type structure.
      In(Type) * T(Group)[Type] >> [](auto& _) { return Type << *_[Type]; },
      In(Type) * T(List)[TypeTuple] >>
        [](auto& _) { return TypeTuple << *_[TypeTuple]; },
      In(Type) * T(Paren)[Type] >> [](auto& _) { return Type << *_[Type]; },

      In(TypeTuple) * T(Group)[Type] >>
        [](auto& _) { return Type << *_[Type]; },

      // Expression structure.
      InExpr * T(Group)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      InExpr * T(List)[Tuple] >> [](auto& _) { return Tuple << *_[Tuple]; },
      InExpr * T(Equals)[Assign] >>
        [](auto& _) { return Assign << *_[Assign]; },
      InExpr * T(Paren)[Expr] >> [](auto& _) { return Expr << *_[Expr]; },
      T(Expr) << (T(Tuple)[Tuple] * End) >> [](auto& _) { return _(Tuple); },

      TypeOrExpr * T(Square)[Typeargs] >>
        [](auto& _) { return Typeargs << *_[Typeargs]; },
      T(Typeargs) << T(List)[Typeargs] >>
        [](auto& _) { return Typeargs << *_[Typeargs]; },
      In(Typeargs) * T(Group)[Type] >> [](auto& _) { return Type << *_[Type]; },
      In(Typeargs) * T(Paren)[Type] >> [](auto& _) { return Type << *_[Type]; },

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
        [](auto& _) { return _(id = Var); },

      // Let.
      In(Expr) * Start * T(Let) * T(Ident)[id] >>
        [](auto& _) { return _(id = Let); },

      // Throw.
      In(Expr) * Start * T(Throw) * (Any++)[rhs] >>
        [](auto& _) { return Throw << (Expr << _[rhs]); },
    };
  }

  Pass references()
  {
    return {
      dir::bottomup,
      {
        // Identifiers and symbols.
        In(Expr) * T(Dot) * Name[id] * ~T(Typeargs)[Typeargs] >>
          [](auto& _) {
            return DotSelector << _[id] << (_[Typeargs] | Typeargs);
          },

        TypeOrExpr * (Name[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](auto& _) {
            auto def = look(_[Type]);
            return reftype(def) << _[id] << (_[Typeargs] | Typeargs);
          },

        // Scoped lookup.
        TypeOrExpr *
            ((T(RefClass) / T(RefType) / T(RefTypeparam) / T(Package))[lhs] *
             T(DoubleColon) * Name[id] * ~T(Typeargs)[Typeargs])[Type] >>
          [](auto& _) {
            auto def = look(_[Type]);
            return reftype(def) << _[lhs] << _[id] << (_[Typeargs] | Typeargs);
          },

        // Use.
        T(Use)[lhs]
            << (T(Type)
                << (T(RefClass) / T(RefType) / T(RefTypeparam) /
                    T(Package))[rhs]) >>
          [](auto& _) {
            auto site = Include ^ _(lhs);
            _.include(site, look(_[rhs]));
            return site << _[rhs];
          },

        // Create sugar.
        In(Expr) * (T(RefClass) / T(RefTypeparam))[lhs] >>
          [](auto& _) {
            return Call << (RefFunction << _[lhs] << Ident(create) << Typeargs)
                        << Expr;
          },
      }};
  }

  Pass typeassert()
  {
    return {
      // Type assertions for operators.
      T(Expr) << (Operator[op] * T(Type)[Type] * End) >>
        [](auto& _) { return _(op) << _[Type]; },
    };
  }

  Pass reverseapp()
  {
    return {
      // Dot: reverse application.
      In(Expr) * Object[lhs] * T(Dot) * Any[rhs] >>
        [](auto& _) { return Call << _[rhs] << _[lhs]; },

      In(Expr) * Object[lhs] * T(DotSelector)[rhs] >>
        [](auto& _) { return Call << (Selector << *_[rhs]) << _[lhs]; },
    };
  }

  Pass application()
  {
    return {
      // Type expressions.
      In(Type) * TypeElem[lhs] * T(Symbol, "~>") * TypeElem[rhs] >>
        [](auto& _) { return TypeView << _[lhs] << _[rhs]; },
      In(Type) * TypeElem[lhs] * T(Symbol, "->") * TypeElem[rhs] >>
        [](auto& _) { return TypeFunc << _[lhs] << _[rhs]; },
      In(Type) * TypeElem[lhs] * T(Symbol, "&") * TypeElem[rhs] >>
        [](auto& _) { return TypeIsect << _[lhs] << _[rhs]; },
      In(Type) * TypeElem[lhs] * T(Symbol, "\\|") * TypeElem[rhs] >>
        [](auto& _) { return TypeUnion << _[lhs] << _[rhs]; },
      In(Type) * T(Throw) * TypeElem[rhs] >>
        [](auto& _) { return TypeThrow << _[rhs]; },

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

      // Type assertions.
      T(Expr) << (Object[lhs] * T(Type)[rhs] * End) >>
        [](auto& _) { return Oftype << _[lhs] << _[rhs]; },
    };
  }

  Pass compaction()
  {
    return {
      // Expression compaction.
      InExpr * T(Expr) << (Any[lhs] * End) >> [](auto& _) { return _(lhs); },
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
        {"references", references()},
        {"typeassert", typeassert()},
        {"compaction", compaction()},
        {"reverseapp", reverseapp()},
        {"application", application()},
        {"compaction", compaction()},
      });

    return d;
  }
}
