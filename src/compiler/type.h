// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/variable_renaming.h"
#include "compiler/region.h"
#include "ds/helpers.h"

#include <cassert>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace verona::compiler
{
  struct Entity;
  struct TypeParameterDef;
  class Context;
  class TypeInterner;

  /*
   * Must be allocated inside a shared_ptr<> by the interner.
   *
   * The "main" constructor of each subclass should be private, with the
   * interner a friend. Unfortunately we have to leave the copy constructors
   * public, as they are called via make_shared in intern.cc
   */
  struct Type : public std::enable_shared_from_this<Type>
  {
    virtual ~Type() {}

    template<typename T>
    std::shared_ptr<const T> dyncast() const
    {
      return std::dynamic_pointer_cast<const T>(shared_from_this());
    }

  protected:
    Type() = default;
    Type(const Type&) = default;

    Type& operator=(const Type&) = delete;
  };
  typedef std::shared_ptr<const Type> TypePtr;
  typedef std::vector<TypePtr> TypeList;
  typedef std::set<TypePtr> TypeSet;

  struct CapabilityType final : public Type
  {
    CapabilityKind kind;
    Region region;

    bool operator<(const CapabilityType& other) const
    {
      return std::tie(kind, region) < std::tie(other.kind, other.region);
    }

  private:
    CapabilityType(CapabilityKind kind, Region region)
    : kind(kind), region(region)
    {
      switch (kind)
      {
        case CapabilityKind::Mutable:
          assert(!std::holds_alternative<RegionNone>(region));
          break;

        case CapabilityKind::Subregion:
          assert(
            !std::holds_alternative<RegionNone>(region) &&
            !std::holds_alternative<RegionHole>(region));
          break;

        case CapabilityKind::Immutable:
          assert(std::holds_alternative<RegionNone>(region));
          break;

        case CapabilityKind::Isolated:
          break;

          EXHAUSTIVE_SWITCH;
      }
    }

    friend TypeInterner;
  };
  typedef std::shared_ptr<const CapabilityType> CapabilityTypePtr;

  struct ApplyRegionType : public Type
  {
    enum Mode
    {
      Adapt,
      Under,
      Extract
    };

    Mode mode;
    Region region;
    TypePtr type;

    bool operator<(const ApplyRegionType& other) const
    {
      return std::tie(mode, region, type) <
        std::tie(other.mode, other.region, other.type);
    }

  private:
    ApplyRegionType(Mode mode, Region region, TypePtr type)
    : mode(mode), region(region), type(type)
    {
      assert(
        !std::holds_alternative<RegionNone>(region) &&
        !std::holds_alternative<RegionHole>(region));
    }
    friend TypeInterner;
  };
  typedef std::shared_ptr<const ApplyRegionType> ApplyRegionTypePtr;

  struct UnapplyRegionType : public Type
  {
    TypePtr type;

    bool operator<(const UnapplyRegionType& other) const
    {
      return type < other.type;
    }

  private:
    UnapplyRegionType(TypePtr type) : type(type) {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const UnapplyRegionType> UnapplyRegionTypePtr;

  struct StaticType final : public Type
  {
    const Entity* definition;
    TypeList arguments;

    bool operator<(const StaticType& other) const
    {
      return std::tie(definition, arguments) <
        std::tie(other.definition, other.arguments);
    }

  private:
    StaticType(const Entity* definition, TypeList arguments)
    : definition(definition), arguments(arguments)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const StaticType> StaticTypePtr;

  struct EntityType final : public Type
  {
    const Entity* definition = nullptr;
    TypeList arguments;

    bool operator<(const EntityType& other) const
    {
      return std::tie(arguments, definition) <
        std::tie(other.arguments, other.definition);
    }

  private:
    EntityType(const Entity* definition, TypeList arguments)
    : arguments(arguments), definition(definition)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const EntityType> EntityTypePtr;

  struct TypeParameter : public Type
  {
    /**
     * During polarization, type parameters are expanded from X to `X & Δ(X)`.
     * To avoid repeatedly expanding X in the result, and make polarization
     * idempotent, we annotate the type parameter with the fact that it has
     * already been expanded.
     */
    enum class Expanded
    {
      Yes,
      No
    };

    const TypeParameterDef* definition = nullptr;
    Expanded expanded;

    bool operator<(const TypeParameter& other) const
    {
      return std::tie(definition, expanded) <
        std::tie(other.definition, other.expanded);
    }

  private:
    TypeParameter(const TypeParameterDef* definition, Expanded expanded)
    : definition(definition), expanded(expanded)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const TypeParameter> TypeParameterPtr;
  typedef std::set<TypeParameterPtr> TypeParameterSet;

  struct ViewpointType : public Type
  {
    std::optional<CapabilityKind> capability;
    TypeParameterSet variables;
    TypePtr right;

    bool operator<(const ViewpointType& other) const
    {
      return std::tie(capability, variables, right) <
        std::tie(other.capability, other.variables, other.right);
    }

  private:
    ViewpointType(
      std::optional<CapabilityKind> capability,
      TypeParameterSet variables,
      TypePtr right)
    : capability(capability), variables(variables), right(right)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const ViewpointType> ViewpointTypePtr;

  struct IntersectionType;
  struct UnionType;
  typedef std::shared_ptr<const UnionType> UnionTypePtr;
  typedef std::shared_ptr<const IntersectionType> IntersectionTypePtr;

  struct UnionType final : public Type
  {
    TypeSet elements;

    bool operator<(const UnionType& other) const
    {
      return elements < other.elements;
    }

    typedef IntersectionType Dual;
    typedef IntersectionTypePtr DualPtr;

  private:
    UnionType(TypeSet elements) : elements(elements) {}
    friend TypeInterner;
  };

  struct IntersectionType final : public Type
  {
    TypeSet elements;

    bool operator<(const IntersectionType& other) const
    {
      return elements < other.elements;
    }

    typedef UnionType Dual;
    typedef UnionTypePtr DualPtr;

  private:
    IntersectionType(TypeSet elements) : elements(elements) {}
    friend TypeInterner;
  };

  enum class Polarity
  {
    Positive,
    Negative,
  };

  struct InferType final : public Type
  {
    uint64_t index;
    std::optional<uint64_t> subindex;
    Polarity polarity;

    bool operator<(const InferType& other) const
    {
      return std::tie(this->index, this->subindex, this->polarity) <
        std::tie(other.index, other.subindex, other.polarity);
    }

  private:
    InferType(
      uint64_t index, std::optional<uint64_t> subindex, Polarity polarity)
    : index(index), subindex(subindex), polarity(polarity)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const InferType> InferTypePtr;
  typedef std::set<InferTypePtr> InferTypeSet;

  /**
   * A range represent any type T where lower <: T and T :< upper.
   * This is only used during type inference, see infer.cc for more details.
   */
  struct RangeType final : public Type
  {
    TypePtr lower;
    TypePtr upper;

    bool operator<(const RangeType& other) const
    {
      return std::tie(lower, upper) < std::tie(other.lower, other.upper);
    }

  private:
    RangeType(TypePtr lower, TypePtr upper) : lower(lower), upper(upper) {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const RangeType> RangeTypePtr;

  struct UnitType final : public Type
  {
    bool operator<(const UnitType& other) const
    {
      return false;
    }

  private:
    UnitType() {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const UnitType> UnitTypePtr;

  struct HasFieldType final : public Type
  {
    TypePtr view;

    std::string name;
    TypePtr read_type;
    TypePtr write_type;

    bool operator<(const HasFieldType& other) const
    {
      return std::tie(view, name, read_type, write_type) <
        std::tie(other.view, other.name, other.read_type, other.write_type);
    }

  private:
    HasFieldType(
      TypePtr view, std::string name, TypePtr read_type, TypePtr write_type)
    : view(view), name(name), read_type(read_type), write_type(write_type)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const HasFieldType> HasFieldTypePtr;

  struct DelayedFieldViewType final : public Type
  {
    std::string name;
    TypePtr type;

    bool operator<(const DelayedFieldViewType& other) const
    {
      return std::tie(name, type) < std::tie(other.name, other.type);
    }

  private:
    DelayedFieldViewType(std::string name, TypePtr type)
    : name(name), type(type)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const DelayedFieldViewType> DelayedFieldViewTypePtr;

  struct BoundedTypeSequence
  {
    TypeList types;

    explicit BoundedTypeSequence(TypeList types) : types(types) {}
    bool operator<(const BoundedTypeSequence& other) const
    {
      return types < other.types;
    }
  };
  struct UnboundedTypeSequence
  {
    uint64_t index;

    explicit UnboundedTypeSequence(uint64_t index) : index(index) {}
    bool operator<(const UnboundedTypeSequence& other) const
    {
      return index < other.index;
    }
  };
  typedef std::variant<BoundedTypeSequence, UnboundedTypeSequence>
    InferableTypeSequence;

  /**
   * Type signature of a method.
   */
  struct TypeSignature
  {
    TypePtr receiver;
    TypeList arguments;
    TypePtr return_type;

    TypeSignature() {}
    TypeSignature(TypePtr receiver, TypeList arguments, TypePtr return_type)
    : receiver(receiver), arguments(arguments), return_type(return_type)
    {}

    bool operator<(const TypeSignature& other) const
    {
      return std::tie(receiver, arguments, return_type) <
        std::tie(other.receiver, other.arguments, other.return_type);
    }
  };

  /**
   * Has a method with the given signature.
   */
  struct HasMethodType final : public Type
  {
    std::string name;
    TypeSignature signature;

    bool operator<(const HasMethodType& other) const
    {
      return std::tie(name, signature) < std::tie(other.name, other.signature);
    }

  private:
    HasMethodType(std::string name, TypeSignature signature)
    : name(name), signature(signature)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const HasMethodType> HasMethodTypePtr;

  /**
   * This has one method which when applied to the given arguments results in
   * the following signature.
   *
   * This differs from HasMethodType in that the type arguments are already
   * applied.
   */
  struct HasAppliedMethodType final : public Type
  {
    std::string name;
    InferableTypeSequence application;
    TypeSignature signature;

    bool operator<(const HasAppliedMethodType& other) const
    {
      return std::tie(name, application, signature) <
        std::tie(other.name, other.application, other.signature);
    }

  private:
    HasAppliedMethodType(
      std::string name,
      InferableTypeSequence application,
      TypeSignature signature)
    : name(name), application(application), signature(signature)
    {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const HasAppliedMethodType> HasAppliedMethodTypePtr;

  struct IsEntityType final : public Type
  {
    bool operator<(const IsEntityType& other) const
    {
      return false;
    }

  private:
    IsEntityType() {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const IsEntityType> IsEntityTypePtr;

  struct FixpointType : public Type
  {
    TypePtr inner;

    bool operator<(const FixpointType& other) const
    {
      return inner < other.inner;
    }

  private:
    FixpointType(TypePtr inner) : inner(inner) {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const FixpointType> FixpointTypePtr;

  struct FixpointVariableType : public Type
  {
    uint64_t depth;
    bool operator<(const FixpointVariableType& other) const
    {
      return depth < other.depth;
    }

  private:
    FixpointVariableType(uint64_t depth) : depth(depth) {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const FixpointVariableType> FixpointVariableTypePtr;

  struct EntityOfType : public Type
  {
    TypePtr inner;

    bool operator<(const EntityOfType& other) const
    {
      return inner < other.inner;
    }

  private:
    EntityOfType(TypePtr inner) : inner(inner) {}
    friend TypeInterner;
  };
  typedef std::shared_ptr<const EntityOfType> EntityOfTypePtr;

  struct VariableRenamingType : public Type
  {
    VariableRenaming renaming;
    TypePtr type;

    bool operator<(const VariableRenamingType& other) const
    {
      return std::tie(renaming, type) < std::tie(other.renaming, other.type);
    }

  private:
    VariableRenamingType(VariableRenaming renaming, TypePtr type)
    : renaming(renaming), type(type)
    {}

    friend TypeInterner;
  };
  typedef std::shared_ptr<const VariableRenamingType> VariableRenamingTypePtr;

  typedef std::map<Variable, TypePtr> PathCompressionMap;
  struct PathCompressionType : public Type
  {
    PathCompressionMap compression;
    TypePtr type;

    bool operator<(const PathCompressionType& other) const
    {
      return std::tie(compression, type) <
        std::tie(other.compression, other.type);
    }

  private:
    PathCompressionType(PathCompressionMap compression, TypePtr type)
    : compression(compression), type(type)
    {}

    friend TypeInterner;
  };
  typedef std::shared_ptr<const PathCompressionType> PathCompressionTypePtr;

  struct IndirectType : public Type
  {
    const BasicBlock* block;
    Variable variable;

    bool operator<(const IndirectType& other) const
    {
      return std::tie(block, variable) < std::tie(other.block, other.variable);
    }

  private:
    IndirectType(const BasicBlock* block, Variable variable)
    : block(block), variable(variable)
    {}

    friend TypeInterner;
  };
  typedef std::shared_ptr<const IndirectType> IndirectTypePtr;

  struct NotChildOfType : public Type
  {
    Region region;

    bool operator<(const NotChildOfType& other) const
    {
      return region < other.region;
    }

  private:
    NotChildOfType(Region region) : region(region)
    {
      assert(std::holds_alternative<RegionVariable>(region));
    }

    friend TypeInterner;
  };
  typedef std::shared_ptr<const NotChildOfType> NotChildOfTypePtr;

  Polarity reverse_polarity(Polarity polarity);
  InferTypePtr reverse_polarity(const InferTypePtr& ty, Context& context);
};
