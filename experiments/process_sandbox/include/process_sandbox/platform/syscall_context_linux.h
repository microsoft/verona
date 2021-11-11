// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#ifdef __linux__
#  include "../callback_numbers.h"

#  include <sys/syscall.h>
#  include <ucontext.h>
namespace sandbox::platform
{
#  ifdef __x86_64__
  /**
   * Linux system call frame inspection.
   */
  class SyscallFrameLinuxX8664
  {
    /**
     * The siginfo structure passed to the signal handler.
     */
    siginfo_t& info;

    /**
     * The context structure containing the complete register contents where
     * the system call was delivered.
     */
    ucontext_t& ctx;

    /**
     * Linux system call numbers.  These must be kept in the same order as
     * the callback numbers.  Each entry is a pair of the callback number
     * followed by the corresponding system-call number.  This is accessed only
     * by a wrapper that provides a compile-tine check that the entries are in
     * the correct order.
     */
    static constexpr std::
      array<std::pair<CallbackKind, int>, SyscallCallbackCount>
        SyscallNumbers{
          std::make_pair(Open, __NR_open),
          {Stat, __NR_stat},
          {Access, __NR_access},
          {OpenAt, __NR_openat},
          {Bind, __NR_bind},
          {Connect, __NR_connect},
        };

  public:
    /**
     * Compile-time lookup of a system call number that corresponds to a given
     * callback.  May return -1 if there is no matching system call.
     */
    template<CallbackKind K>
    static constexpr int syscall_number()
    {
      static_assert(
        K < SyscallNumbers.size(), "Callback number is out of range");
      static_assert(
        SyscallNumbers[K].first == K,
        "SyscallNumbers array layout is incorrect!");
      return SyscallNumbers[K].second;
    }

    /**
     * The signal sent when a seccomp policy violation occurs.
     */
    static constexpr int syscall_signal = SIGSYS;

    /**
     * Constructor.  Takes the arguments to a signal handler.
     */
    SyscallFrameLinuxX8664(siginfo_t& i, ucontext_t& c) : info(i), ctx(c) {}

    /**
     * Get the syscall argument at index `Arg`, cast to type `T`.
     */
    template<int Arg, typename T = uintptr_t>
    T get_arg()
    {
      static_assert(Arg < 6, "Unable to access more than 6 arguments");
      static_assert(
        std::is_integral_v<T> || std::is_pointer_v<T>,
        "Syscall arguments can only be accessed as integers or pointers");
      auto cast = [](register_t val) {
        if constexpr (std::is_integral_v<T>)
        {
          return static_cast<T>(val);
        }
        if constexpr (std::is_pointer_v<T>)
        {
          return reinterpret_cast<T>(static_cast<intptr_t>(val));
        }
      };
      switch (Arg)
      {
        case 0:
          return cast(ctx.uc_mcontext.gregs[REG_RDI]);
        case 1:
          return cast(ctx.uc_mcontext.gregs[REG_RSI]);
        case 2:
          return cast(ctx.uc_mcontext.gregs[REG_RDX]);
        case 3:
          return cast(ctx.uc_mcontext.gregs[REG_R10]);
        case 4:
          return cast(ctx.uc_mcontext.gregs[REG_R8]);
        case 5:
          return cast(ctx.uc_mcontext.gregs[REG_R9]);
      }
    }

    /**
     * Set a return value for a successful system call result.  On Linux, this
     * returned in the normal return register as a positive integer.
     *
     * Also advance the instruction pointer to after the system call.
     */
    void set_success_return(intptr_t r)
    {
      ctx.uc_mcontext.gregs[REG_RAX] = static_cast<greg_t>(r);
      // Advance the PC past the syscall instruction.
      ctx.uc_mcontext.gregs[REG_RIP] = (greg_t)info.si_call_addr;
    }

    /**
     * Set a return value for an unsuccessful system call result.  On Linux,
     * this returned in the normal return register as an integer.
     *
     * Also advance the instruction pointer to after the system call.
     */
    void set_error_return(int e)
    {
      // Linux uses negated values to indicate errno returns.
      ctx.uc_mcontext.gregs[REG_RAX] = 0 - static_cast<greg_t>(e);
      // Advance the PC past the syscall instruction.
      ctx.uc_mcontext.gregs[REG_RIP] = (greg_t)info.si_call_addr;
    }

    /**
     * Get the system call number.
     */
    int get_syscall_number()
    {
      return info.si_syscall;
    }

    /**
     * Is this a signal delivered as a result of a seccomp policy violation?
     */
    bool is_sandbox_policy_violation()
    {
      // This ought to be SYS_SECCOMP, but including the header that defines it
      // causes conflicts with other Linux headers.
      return info.si_code == 1;
    }
  };

  /**
   * Export the Linux type for this architecture.
   */
  using SyscallFrameLinux = SyscallFrameLinuxX8664;
#  else
#    error Your architecture is not yet supported
#  endif
}
#endif
