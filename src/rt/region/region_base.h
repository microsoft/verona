// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "externalreference.h"
#include "rememberedset.h"

namespace verona::rt
{
  using namespace snmalloc;

  /**
   * Please see region.h for the full documentation.
   *
   * This is the base class for concrete region implementations, and contains
   * all the common functionality. Because of difficulties with dependencies,
   * this class is intentionally minimal and contains no helpers---it is not
   * aware of any of the concrete region implementation classes.
   **/

  enum class RegionType
  {
    Trace,
    Arena,
  };

  class RegionBase : public Object,
                     public ExternalReferenceTable,
                     public RememberedSet
  {
    friend class Freeze;
    friend class RegionTrace;
    friend class RegionArena;

  public:
    enum IteratorType
    {
      Trivial,
      NonTrivial,
      AllObjects,
    };

    RegionBase() : Object() {}

  private:
    inline void dealloc(Alloc* alloc)
    {
      ExternalReferenceTable::dealloc(alloc);
      RememberedSet::dealloc(alloc);
      Object::dealloc(alloc);
    }
  };

} // namespace verona::rt