// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * This file contains the definition of a per-platform sandboxing policy.
 * This interface is probably not yet sufficiently flexible and so is likely to
 * change.  It is currently used with Capsicum and seccomp-bpf.
 *
 * Implementations of this should implement two static methods.
 * The first is responsible for launching the process:
 *
 * ```c++
 * template<size_t EnvSize, size_t LibDirSize>
 * static void execve(
 *   const char* pathname,
 *   const std::array<const char*, EnvSize> &envp,
 *   const std::array<const char*, LibDirSize> &libdirs)
 * ```
 *
 * This takes the path to the `library_runner` binary, the environment
 * variables (including a null terminator) and the paths that can be used for
 * loading shared libraries.  This method should not return, if it fails to
 * launch the library runner then it must exit.  This method is always called
 * from a fork-like child process and so exiting will not kill the parent
 * process.
 *
 * The second method applies any parts of the policy that must be applied after
 * the library runner has launched, but before it has loaded any untrusted
 * code.
 *
 * ```c++
 * static void apply_sandboxing_policy_postexec();
 * ```
 *
 * Note that some implementations may enforce all sandboxing policy in one of
 * these methods.  If possible, applying the entire policy pre-exec is
 * preferred.
 */

namespace sandbox::platform
{
  /**
   * Trivial sandboxing policy, does not enforce any sandboxing.  This can be
   * used for lightweight fault isolation on any POSIX platform, but does not
   * restrict the child process's access to the global namespace.
   * Memory-safety bugs in the child will not impact the parent directly but
   * a vulnerability in the child can be exploited to gain ambient authority.
   */
  struct SandboxNoOp
  {
    /**
     * Do nothing: no policy is enforced after exec.
     */
    static void apply_sandboxing_policy_postexec() {}

    /**
     * Execute the library runner.
     */
    template<size_t EnvSize, size_t LibDirSize>
    static void execve(
      const char* pathname,
      const std::array<const char*, EnvSize>& envp,
      const std::array<const char*, LibDirSize>&)
    {
      char* args[] = {const_cast<char*>(pathname), nullptr};
      ::execve(pathname, args, const_cast<char**>(envp.data()));
      SANDBOX_INVARIANT(0, "Execve failed: {}", strerror(errno));
    }
  };
}

#include "sandbox_capsicum.h"
#include "sandbox_seccomp-bpf.h"

namespace sandbox::platform
{
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
