// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
/**
 * Child process abstraction layer.  This currently abstracts only over
 * POSIX-like systems with fork-like process creation.  On Windows-like
 * systems, much of the logic for configuring the resources for the child
 * process will need to be different and at that point this will be sunk into a
 * lower-level platform abstraction.
 *
 * This exposes a class with the following interface:
 *
 * ```c++
 * class ChildProcess
 * {
 *   template<typename T>
 *   ChildProcess(T&& start);
 *   ExitStatus exit_status()
 *   ExitStatus wait_for_exit()
 * };
 * ```
 *
 * The constructor takes a callable object that is executed in the context of
 * the callee.  In some implementations, this will share a heap with the parent
 * and so must free any memory that it allocates.
 *
 * The `exit_status` method polls whether the child has exited and returns the
 * error code if it has.
 *
 * The `wait_for_exit` method is a blocking call that does not return until the
 * child process has exited and then returns the same value as `exit_status`.
 *
 * FIXME: Currently, `wait_for_exit` allows a malicious child to cause the
 * parent to block forever.  This will eventually be modified with a timeout
 * and allow the caller to handle cases where the child process is taking an
 * unreasonable amount of time (though the definition of 'unreasonable' is very
 * subjective and use dependent, so a lot more work is required to understand
 * what this really means).
 */

namespace sandbox
{
  namespace platform
  {
    /**
     * The exit status for a child process.  The child process may not have
     * exited yet, this structure captures both whether the process has exited
     * and the exit code if it has.
     */
    struct ExitStatus
    {
      /**
       * Predicate indicating whether the process has exited.
       */
      bool has_exited = false;
      /**
       * The exit code.  If `has_exited` is false, then the value of this is
       * undefined.
       */
      int exit_code = 0;
    };
  }
}

#include "child_process_pdfork.h"
#include "child_process_vfork.h"

namespace sandbox
{
  namespace platform
  {
    using ChildProcess =
#if defined(USE_KQUEUE_PROCDESC)
      ChildProcessPDFork
#elif defined(__unix__)
      ChildProcessVFork
#else
#  error No child process creation class defined
#endif
      ;
  }
}
