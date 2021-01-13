// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This file contains the definition of a per-platform sandboxing policy.
 * Because this interface currently has only one implementation, it is almost
 * certainly the wrong abstraction and will change when others (e.g.
 * seccomp-bpf) are added.
 */

#include "sandbox_capsicum.h"
#include "sandbox_seccomp-bpf.h"

namespace sandbox
{
  namespace platform
  {
    struct SandboxNoOp
    {
      template<typename T, typename U>
      void restrict_file_descriptors(const T&, const U&)
      {}

      static void apply_sandboxing_policy_postexec() {}

      void apply_sandboxing_policy_preexec() {}
    };

    using Sandbox =
#ifdef USE_CAPSICUM
      SandboxCapsicum
#elif defined(__linux__)
      SandboxSeccompBPF
#else
      SandboxNoOp
#endif
      ;
  }
}
