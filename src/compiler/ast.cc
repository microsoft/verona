// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "compiler/ast.h"

#include "compiler/format.h"
#include "compiler/instantiation.h"
#include "compiler/intern.h"
#include "compiler/printing.h"
#include "compiler/substitution.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  using pegmatite::ASTConstant;
  using pegmatite::ASTStack;
  using pegmatite::ASTStackEntry;
  using pegmatite::ErrorReporter;
  using pegmatite::InputRange;
  using pegmatite::popFromASTStack;

  namespace
  {
    /**
     * Get a string describing the type arguments for given generics and
     * instantiation.
     *
     * The string will be empty if there are no generics, or will be of the form
     * "[T1, T2]" otherwise.
     */
    std::string instantiated_generics(
      const Generics& generics, const Instantiation& instantiation)
    {
      using format::optional_list;
      using format::to_string;

      auto map_parameter = [&](const std::unique_ptr<TypeParameterDef>& elem) {
        return instantiation.types().at(elem->index);
      };

      return to_string(optional_list(generics.types, map_parameter));
    }
  }

  const Name& Symbol::name() const
  {
    return match(
      *this,
      [](const ErrorSymbol&) -> const Name& {
        throw std::logic_error("Found error symbol");
      },
      [](const auto* inner) -> const Name& { return inner->name; });
  }

  std::string Entity::path() const
  {
    return name;
  }

  std::string
  Entity::instantiated_path(const Instantiation& instantiation) const
  {
    return fmt::format(
      "{}{}", name, instantiated_generics(*generics, instantiation));
  }

  std::string Member::path() const
  {
    if (parent == nullptr)
      throw std::logic_error("path called before resolution");

    return parent->path() + "." + name;
  }

  std::string
  Method::instantiated_path(const Instantiation& instantiation) const
  {
    if (parent == nullptr)
      throw std::logic_error("instantiated_path called before resolution");

    return fmt::format(
      "{}.{}{}",
      parent->instantiated_path(instantiation),
      name,
      instantiated_generics(*signature->generics, instantiation));
  }

  std::string Field::instantiated_path(const Instantiation& instantiation) const
  {
    if (parent == nullptr)
      throw std::logic_error("instantiated_path called before resolution");

    std::string buf;
    buf.append(parent->instantiated_path(instantiation));
    buf.append(".");
    buf.append(name);
    return buf;
  }

  std::string_view binary_operator_method_name(BinaryOperator op)
  {
    switch (op)
    {
      case BinaryOperator::Add:
        return "u64_add";
      case BinaryOperator::Sub:
        return "u64_sub";
      case BinaryOperator::Lt:
        return "u64_lt";
      case BinaryOperator::Le:
        return "u64_le";
      case BinaryOperator::Gt:
        return "u64_gt";
      case BinaryOperator::Ge:
        return "u64_ge";
      case BinaryOperator::Eq:
        return "u64_eq";
      case BinaryOperator::Ne:
        return "u64_ne";
      case BinaryOperator::And:
        return "u64_and";
      case BinaryOperator::Or:
        return "u64_or";

        EXHAUSTIVE_SWITCH
    }
  }
}
