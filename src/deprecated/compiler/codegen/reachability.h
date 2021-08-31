// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/codegen/generator.h"
#include "compiler/codegen/selector.h"
#include "compiler/instantiation.h"

#include <unordered_set>

/**
 * The reachability phase determines the set of items that need to be
 * codegenerated.
 *
 * It also assigns them relocatable labels and descriptor identifiers. These
 * will be defined to the correct value when they are emitted.
 */
namespace verona::compiler
{
  struct AnalysisResults;

  /**
   * A reference to an AST node with a specific instantiation.
   */
  template<typename T>
  struct CodegenItem
  {
    const T* definition;
    Instantiation instantiation;

    explicit CodegenItem(const T* definition, Instantiation instantiation)
    : definition(definition), instantiation(instantiation)
    {}

    std::string instantiated_path() const
    {
      return definition->instantiated_path(instantiation);
    }

    bool operator<(const CodegenItem<T>& other) const
    {
      return std::tie(definition, instantiation) <
        std::tie(other.definition, other.instantiation);
    }

    bool operator==(const CodegenItem<T>& other) const
    {
      return std::tie(definition, instantiation) ==
        std::tie(other.definition, other.instantiation);
    }
  };

  struct MethodReachability
  {
    MethodReachability(std::optional<Label> label) : label(label) {}

    // For methods without a body, this is nullopt.
    std::optional<Label> label;
  };

  struct EntityReachability
  {
    EntityReachability(Descriptor descriptor)
    : descriptor(descriptor), finaliser(std::nullopt)
    {}

    Descriptor descriptor;
    std::map<CodegenItem<Method>, MethodReachability> methods;

    MethodReachability finaliser;

    // Set of reified entities which are subtypes of this one. It will only be
    // non-empty for interfaces.
    std::set<CodegenItem<Entity>> subtypes;
  };

  struct Reachability
  {
    std::map<CodegenItem<Entity>, EntityReachability> entities;
    std::set<Selector> selectors;

    /**
     * There can be multiple equivalent entities that are reachable from the
     * program. In this case we pick a canonical one (the first one we come
     * across) and make the others point to it in this map.
     *
     * Only the canonical entity will be included in the program.
     */
    std::map<CodegenItem<Entity>, CodegenItem<Entity>> equivalent_entities;

    /**
     * Find the canonical item that is equivalent to `entity`.
     */
    const CodegenItem<Entity>&
    normalize_equivalence(const CodegenItem<Entity>& entity) const;

    /**
     * Find the information related to this entity or an equivalent one.
     */
    EntityReachability& find_entity(const CodegenItem<Entity>& entity);
    const EntityReachability&
    find_entity(const CodegenItem<Entity>& entity) const;

    /**
     * Find the information related to this entity or an equivalent one,
     * returns nullptr if the item is not reachable.
     */
    const EntityReachability*
    try_find_entity(const CodegenItem<Entity>& entity) const;
  };

  Reachability compute_reachability(
    Context& context,
    const Program& program,
    Generator& gen,
    CodegenItem<Entity> main_class,
    CodegenItem<Method> main_method,
    const AnalysisResults& analysis);

  std::ostream& operator<<(std::ostream& s, const CodegenItem<Method>& item);
  std::ostream& operator<<(std::ostream& s, const CodegenItem<Entity>& item);
  std::ostream& operator<<(std::ostream& s, const Selector& selector);
}
