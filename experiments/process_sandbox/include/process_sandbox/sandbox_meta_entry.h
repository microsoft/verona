// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <snmalloc/snmalloc_core.h>

namespace sandbox
{
  /**
   * Pagemap entry.  Extends the front-end's version to use one bit to identify
   * pagemap entries as owned by the child.
   */
  class SandboxMetaEntry
  : public snmalloc::FrontendMetaEntry<snmalloc::FrontendSlabMetadata>
  {
    /**
     * Bit set if this metaentry is owned by the sandbox.
     */
    static constexpr snmalloc::address_t SANDBOX_BIT = 1 << 3;

  public:
    /**
     * Inherit all constructors.
     */
    using snmalloc::FrontendMetaEntry<
      snmalloc::FrontendSlabMetadata>::FrontendMetaEntry;

    /**
     * Does this metaentry correspond to sandbox-owned memory
     */
    bool is_sandbox_owned() const
    {
      return (meta & SANDBOX_BIT) == SANDBOX_BIT;
    }

    /**
     * Claim this entry for the sandbox.
     */
    void claim_for_sandbox()
    {
      meta |= SANDBOX_BIT;
    }

    [[nodiscard]] bool is_unowned() const
    {
      auto m = meta & ~SANDBOX_BIT;
      return ((m == 0) || (m == META_BOUNDARY_BIT)) &&
        (remote_and_sizeclass == 0);
    }

    [[nodiscard]] SNMALLOC_FAST_PATH snmalloc::FrontendSlabMetadata*
    get_slab_metadata() const
    {
      SNMALLOC_ASSERT(get_remote() != nullptr);
      auto m = meta & ~(SANDBOX_BIT | META_BOUNDARY_BIT);
      return snmalloc::unsafe_from_uintptr<snmalloc::FrontendSlabMetadata>(m);
    }
  };
}
