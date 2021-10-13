// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#ifdef __FreeBSD__
#  include <sys/syscall.h>
#  include <ucontext.h>

namespace sandbox::platform
{
#  ifdef __x86_64__
  /**
   * FreeBSD system call information.
   *
   * Note: This depends on FreeBSD D29185 landing.
   */
  class SyscallFrameFreeBSDX8664
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
     * Is this a `syscall` instruction?  This is a special system call that
     * takes the system call number in the first argument register and so every
     * argument is one later.
     */
    bool is_syscall()
    {
      // This check works only on newer FreeBSD systems that provide si_syscall.
      // Once support for this is MFC'd to all supporting branches and a release
      // has happened from them then the fallback codepaths should all be
      // deleted from here.  At the time of writing, it is in 14, not yet merged
      // back to 12.x or 13.x.
#    ifdef si_syscall
      return (info.si_syscall == SYS_syscall) ||
        (info.si_syscall == SYS___syscall);
#    else
      return false;
#    endif
    }

  public:
    /**
     * FreeBSD system call numbers.
     */
    enum SyscallNumbers
    {
      Open = SYS_open,
      OpenAt = SYS_openat,
      Stat = SYS_freebsd11_stat,
    };

    /**
     * The signal delivered for a Capsicum-disallowed system call.
     */
    static constexpr int syscall_signal = SIGTRAP;

    /**
     * Constructor.  Takes the arguments to a signal handler.
     */
    SyscallFrameFreeBSDX8664(siginfo_t& i, ucontext_t& c) : info(i), ctx(c) {}

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
      switch (Arg + (is_syscall() ? 1 : 0))
      {
        case 0:
          return cast(ctx.uc_mcontext.mc_rdi);
        case 1:
          return cast(ctx.uc_mcontext.mc_rsi);
        case 2:
          return cast(ctx.uc_mcontext.mc_rdx);
        case 3:
          return cast(ctx.uc_mcontext.mc_r10);
        case 4:
          return cast(ctx.uc_mcontext.mc_r8);
        case 5:
          return cast(ctx.uc_mcontext.mc_r9);
      }
      return 0;
    }

    /**
     * Set a successful error return value.  FreeBSD returns this in the first
     * return register.  On x86, it clears the carry flag to indicate a
     * successful return.  This does not need to update the program counter
     * because this is set to the syscall return by Capsicum.
     */
    void set_success_return(intptr_t r)
    {
      // Clear the carry flag to indicate success
      ctx.uc_mcontext.mc_rflags &= ~1;
      // Set the return value
      ctx.uc_mcontext.mc_rax = static_cast<register_t>(r);
    }

    /**
     * Set an unsuccessful error return value.  FreeBSD returns this in the
     * first return register.  On x86, it sets the carry flag to indicate an
     * unsuccessful return.  This does not need to update the program counter
     * because this is set to the syscall return by Capsicum.
     */
    void set_error_return(int e)
    {
      // Indicate that this is an error return by setting the carry flag
      ctx.uc_mcontext.mc_rflags |= 1;
      // Store the errno value in the return register
      ctx.uc_mcontext.mc_rax = static_cast<register_t>(e);
    }

    /**
     * Get the system call number.  This is passed into the signal handler, but
     * if this is a `syscall` system call then we need to access it from the
     * first argument.
     */
    int get_syscall_number()
    {
#    ifdef si_syscall
      return is_syscall() ? ctx.uc_mcontext.mc_rdi : info.si_syscall;
#    else
      return ctx.uc_mcontext.mc_rax;
#    endif
    }

    /**
     * Is this a Capsicum violation?
     */
    bool is_sandbox_policy_violation()
    {
      return info.si_code == TRAP_CAP;
    }
  };

  /**
   * Expose this as the FreeBSD syscall frame handler.
   */
  using SyscallFrameFreeBSD = SyscallFrameFreeBSDX8664;
#  else
#    error Your architecture is not yet supported
#  endif
}
#endif
