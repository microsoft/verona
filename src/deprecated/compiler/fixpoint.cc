// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "compiler/fixpoint.h"

#include "compiler/context.h"
#include "compiler/mapper.h"

namespace verona::compiler
{
  class OpenFixpoint : public RecursiveTypeMapper
  {
  public:
    OpenFixpoint(Context& context, TypePtr value)
    : RecursiveTypeMapper(context), value_(value)
    {}

    TypePtr
    visit_fixpoint_variable_type(const FixpointVariableTypePtr& ty) final
    {
      assert(ty->depth <= depth_);
      if (ty->depth == depth_)
        return value_;
      else
        return ty;
    }

    TypePtr visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      depth_ += 1;
      auto inner = apply(ty->inner);
      depth_ -= 1;
      return context().mk_fixpoint(inner);
    }

    TypePtr value_;
    size_t depth_ = 0;
  };

  class CloseFixpoint : public RecursiveTypeMapper
  {
  public:
    CloseFixpoint(Context& context, InferTypePtr infer)
    : RecursiveTypeMapper(context), infer_(infer), depth_(0)
    {}

  private:
    TypePtr visit_infer(const InferTypePtr& ty) final
    {
      if (ty == infer_)
        return context().mk_fixpoint_variable(0);
      else
        return ty;
    }

    TypePtr visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      depth_ += 1;
      auto inner = apply(ty->inner);
      depth_ -= 1;
      return context().mk_fixpoint(inner);
    }

    InferTypePtr infer_;
    size_t depth_;
  };

  class ShiftFixpoint : public RecursiveTypeMapper
  {
  public:
    ShiftFixpoint(Context& context, ptrdiff_t distance, size_t cutoff)
    : RecursiveTypeMapper(context), distance_(distance), cutoff_(cutoff)
    {}

    TypePtr
    visit_fixpoint_variable_type(const FixpointVariableTypePtr& ty) final
    {
      if (ty->depth < cutoff_)
      {
        return ty;
      }
      else
      {
        assert(ty->depth + distance_ >= 0);
        return context().mk_fixpoint_variable(ty->depth + distance_);
      }
    }

    TypePtr visit_fixpoint_type(const FixpointTypePtr& ty) final
    {
      cutoff_ += 1;
      auto inner = apply(ty->inner);
      cutoff_ -= 1;
      return context().mk_fixpoint(inner);
    }

  private:
    ptrdiff_t distance_;
    size_t cutoff_;
  };

  TypePtr unfold_fixpoint(Context& context, const FixpointTypePtr& fixpoint)
  {
    return OpenFixpoint(context, fixpoint).apply(fixpoint->inner);
  }

  TypePtr close_fixpoint(
    Context& context, const InferTypePtr& infer, const TypePtr& type)
  {
    return CloseFixpoint(context, infer).apply(type);
  }

  TypePtr shift_fixpoint(
    Context& context, const TypePtr& type, ptrdiff_t distance, size_t cutoff)
  {
    return ShiftFixpoint(context, distance, cutoff).apply(type);
  }
}
