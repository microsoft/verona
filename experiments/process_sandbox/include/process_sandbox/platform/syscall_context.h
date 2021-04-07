// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This file provides helpers for extracting system call arguments from a trap
 * frame delivered into a signal handler and restoring them.  If a sandboxing
 * mechanism provides the option to receive a signal when a sandboxed process
 * tries to make a disallowed system call, we use that mechanism to deliver an
 * upcall. This requires the ability to pull the arguments out of the
 * `ucontext_t` structure and re-inject the return values.
 *
 * Each platform / architecture combination must implement a version a class,
 * with the following members:
 *
 * ```c++
 * SyscallFrame(siginfo_t& i, ucontext_t& c);
 * ```
 *
 * The constructor must take references to the platform's `siginf_t` and
 * `ucontext_t` structures, as they appear in a signal handler.
 *
 * ```c++
 * enum SyscallNumbers;
 * ```
 *
 * This uses the same names as `UpcallKind`, but defines them numerically as
 * the platform's system call number.  If the platform does not have a specific
 * system call, this must be defined as -1.
 *
 * ```c++
 * template<int Arg, typename T = uintptr_t>
 * T get_arg()
 * ```
 *
 * Returns the system call argument at index `Arg`, typed as `T` and should
 * trigger a compile failure if this is not possible.
 *
 * ```c++
 * void set_success_return(intptr_t r)
 * ```
 *
 * Updates the `ucontext_t` as if the system call had returned `r` as a success
 * value.  If the platform delivers traps before the system call instruction,
 * this must also update the program counter to point after the system call
 * instruction.
 *
 * ```c++
 * void set_error_return(int e)
 * ```
 *
 * Updates the `ucontext_t` as if the system call had returned `e` as the value
 * to store in `errno`.  If the platform delivers traps before the system call
 * instruction, this must also update the program counter to point after the
 * system call instruction.
 *
 * ```c++
 * int get_syscall_number()
 * ```
 *
 * Returns the number of the system call delivered in this frame.
 * ```c++
 * bool is_sandbox_policy_violation()
 * ```
 *
 * Returns whether this signal was delivered as the result of a sandbox policy
 * violation.
 */

#include "syscall_context_freebsd.h"
#include "syscall_context_linux.h"

namespace sandbox::platform
{
  using SyscallFrame =
#ifdef __FreeBSD__
    SyscallFrameFreeBSD
#elif defined(__linux__)
    SyscallFrameLinux
#else
#  error Your platform is not yet supported
#endif
    ;
}
