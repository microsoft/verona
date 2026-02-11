#include "../infix.h"

namespace infix {

PassDef get_parse_types_pass() {
  // Normalise type lookups into an explicit chain of `Lookup` nodes, ready for
  // later resolution. This pass rewrites a leading name (with optional type
  // arguments) into a `Lookup`, then repeatedly consumes `::`-separated
  // segments to extend the chain.
  PassDef pass{
      "parse_types",
      wf_function_parse,
      dir::topdown,
      {
          // Start of a type: Name [Square]? -> Lookup(Name, Args).
          In(Type, Use) * Start * ~T(TypeOr)[TypeOr] * T(Name)[Name] * ~T(Square)[Args] >>
              [](auto &_) {
                auto reference = TypeReference << _(Name)
                                               << (+(TypeArgs << +(*_[Args])));
                auto result = TypeLookup << +reference;
                return Seq << _[TypeOr] << result;
              },

          // Extend lookup chain: Lookup :: Name [Square]? ->
          // Lookup Lookup.
          In(Type, Use) * T(TypeLookup)[Lhs] * T(DoubleColon) * T(Name)[Name] *
                  ~T(Square)[Args] >>
              [](auto &_) {
                auto extension = TypeReference << (+_(Name))
                                               << (+(TypeArgs << +(*_[Args])));
                auto result = _(Lhs) << (+extension);
                return result;
              },

          In(TypeArgs) * T(Group)[Group] >>
              [](auto &_) { return Type << (*_(Group)); },

          Start * T(TypeLookup)[Lhs] * T(Name, "\\|")>>
              [](auto &_) { return TypeOr << _(Lhs); },

          T(TypeOr)[Lhs] * T(TypeLookup)[Rhs] * (T(Name, "\\|") / End) >>
              [](auto &_) { return _(Lhs) << _[Rhs]; },

      }};
  return pass;
}

} // namespace infix
