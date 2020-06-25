// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/structural.h"

#include "compiler/resolution.h"
#include "compiler/zip.h"

namespace verona::compiler
{
  namespace
  {
    class StructuralConstraints : public MemberVisitor<TypePtr>
    {
    public:
      StructuralConstraints(
        Context& context, const Instantiation& instantiation)
      : context_(context), instantiation_(instantiation)
      {}

      /**
       * Given the following definition:
       *
       * interface I[X]
       *   f: X
       *
       * For I[A].f, we build the following constraint:
       *
       * { f: (A, A) }
       *
       * The field type apears in both the read and the write position, in order
       * to get invariance.
       */
      TypePtr visit_field(Field* field) final
      {
        // Use mut(.) as a no-op view.
        TypePtr view = context_.mk_mutable(RegionHole{});
        TypePtr field_type = instantiation_.apply(context_, field->type);
        return context_.mk_has_field(view, field->name, field_type, field_type);
      }

      /**
       * Given the following definition:
       *
       * interface I[X]
       *   fun m(u: X): Y;
       *
       * For I[A].m, we build the following constraint:
       *
       * { m: A -> A }
       *
       */
      TypePtr visit_method(Method* method) final
      {
        // TODO: support for generic methods. For now we return bottom as a
        // constraint, which can never be satisified.
        if (method->signature->generics->types.size() > 0)
          return context_.mk_bottom();

        TypeSignature signature =
          instantiation_.apply(context_, method->signature->types);
        return context_.mk_has_method(method->name, signature);
      }

    private:
      Context& context_;
      const Instantiation& instantiation_;
    };

    /**
     * Given two TypeSignatures, adds the relevant sub-constraints to the
     * solution.
     *
     * If the signatures' arities don't match, no sub-constraints are generated
     * and false is returned.
     *
     * Receiver and arguments are contra-variant, return type is co-variant.
     *
     *            T4 <: T1
     *            T5 <: T2
     *            T3 <: T6
     * ------------------------------
     *  (T1, T2): T3 <: (T4, T5): T6
     */
    bool add_signature_constraints(
      Constraint::Compound* solution,
      const TypeSignature& left,
      const TypeSignature& right)
    {
      if (left.arguments.size() != right.arguments.size())
        return false;

      solution->add(right.receiver, left.receiver);
      solution->add_pairwise(right.arguments, left.arguments);
      solution->add(left.return_type, right.return_type);
      return true;
    }

    /**
     * Add the right constraints to enforce that `arguments` sastifies the
     * bounds specified in `generics`.
     */
    void add_bounds_constraints(
      Context& context,
      Constraint::Compound* solution,
      const Instantiation& instantiation,
      const TypeList& arguments,
      const Generics& generics)
    {
      for (const auto& [argument, generic] :
           safe_zip(arguments, generics.types))
      {
        TypePtr bound = instantiation.apply(context, generic->bound);
        solution->add(argument, bound);
      }
    }

    std::optional<TypeList> unify_type_sequence_length_inner(
      Context& context,
      Constraint::Compound* solution,
      const BoundedTypeSequence& sequence,
      size_t length)
    {
      if (sequence.types.size() != length)
        return std::nullopt;
      else
        return sequence.types;
    }

    std::optional<TypeList> unify_type_sequence_length_inner(
      Context& context,
      Constraint::Compound* solution,
      const UnboundedTypeSequence& sequence,
      size_t length)
    {
      TypeList result;
      for (size_t i = 0; i < length; i++)
      {
        result.push_back(context.mk_infer_range(sequence.index, i));
      }
      solution->substitute(sequence, BoundedTypeSequence(result));
      return result;
    }

    /**
     * Unify the length of an InferableTypeSequence, and get the contents of the
     * sequence.
     *
     * If the sequence is unbounded, this will add a substitution in `solution`
     * to assign it a length and fill the sequence with InferType.
     *
     * If the sequence is already bounded but the lengths don't match,
     * std::nullopt is returned. If the lengths do match, the existing contents
     * of the sequence is returned.
     */
    std::optional<TypeList> unify_type_sequence_length(
      Context& context,
      Constraint::Compound* solution,
      const InferableTypeSequence& sequence,
      size_t length)
    {
      return std::visit(
        [&](const auto& inner) {
          return unify_type_sequence_length_inner(
            context, solution, inner, length);
        },
        sequence);
    }
  }

  void add_structural_constraints(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const EntityTypePtr& super)
  {
    assert(super->definition->kind->value() == Entity::Kind::Interface);

    Instantiation instantiation(super->arguments);

    // The LHS must have some entity-like type.
    // This prevents things like capabilities from being subtypes of the empty
    // interface.
    solution->add(sub, context.mk_is_entity());

    StructuralConstraints visitor(context, instantiation);
    for (const auto& member : super->definition->members)
    {
      TypePtr constraint = visitor.visit_member(member.get());
      solution->add(sub, constraint);
    }
  }

  bool solve_has_method(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasMethodTypePtr& super)
  {
    std::optional<std::pair<TypeList, const FnSignature*>> signature =
      lookup_method_signature(sub, super->name);
    if (!signature)
      return false;

    // TODO: support for generic methods.
    if (signature->second->generics->types.size() > 0)
      return false;

    Instantiation instantiation(signature->first);
    TypeSignature left = instantiation.apply(context, signature->second->types);

    if (!add_signature_constraints(solution, left, super->signature))
      return false;

    return true;
  }

  bool solve_has_applied_method(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasAppliedMethodTypePtr& super)
  {
    std::optional<std::pair<TypeList, const FnSignature*>> signature =
      lookup_method_signature(sub, super->name);

    if (!signature)
      return false;

    // HasAppliedMethodType might have a unspecified number of type arguments.
    // We need to unify that sequence length with the expected number of
    // arguments.
    size_t generics_size = signature->second->generics->types.size();
    std::optional<TypeList> application = unify_type_sequence_length(
      context, solution, super->application, generics_size);

    if (!application)
      return false;

    // We need to substitute the method signature using both the type arguments
    // applied to the class (signature->first) and the type arguments applied to
    // the method (*application).
    Instantiation instantiation(signature->first, *application);

    // Make sure the type arguments for the method match the bounds specified by
    // the method.
    add_bounds_constraints(
      context,
      solution,
      instantiation,
      *application,
      *signature->second->generics);

    TypeSignature left = instantiation.apply(context, signature->second->types);

    if (!add_signature_constraints(solution, left, super->signature))
      return false;

    return solution;
  }

  bool solve_has_field(
    Context& context,
    Constraint::Compound* solution,
    const TypePtr& sub,
    const HasFieldTypePtr& super)
  {
    TypeSet elements;
    if (auto isect = sub->dyncast<IntersectionType>())
      elements = isect->elements;
    else
      elements.insert(sub);

    // This will ignore components of `elements` which aren't entities or don't
    // have this field.
    TypeSet field_types = lookup_field_types(context, elements, super->name);

    if (field_types.empty())
      return false;

    solution->add(
      context.mk_viewpoint(super->view, context.mk_intersection(field_types)),
      super->read_type);
    solution->add(
      super->write_type,
      context.mk_viewpoint(super->view, context.mk_union(field_types)));

    return true;
  }
}
