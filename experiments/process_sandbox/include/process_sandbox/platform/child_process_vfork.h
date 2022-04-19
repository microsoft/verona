// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#ifdef __unix__
#  include <err.h>
#  include <errno.h>
#  include <signal.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  if __has_include(<sys/procctl.h>)
#    include <sys/procctl.h>
#  endif
#  include "../helpers.h"
namespace sandbox
{
  namespace platform
  {
    /**
     * Implementation of the child process interface that uses `vfork` to
     * create the child process and `waitpid` to wait for completion.  This
     * implementation should be portable to all POSIX systems.
     */
    class ChildProcessVFork
    {
      /**
       * A handle to a process.  The generic *NIX code path uses a Process ID,
       * which is not actually a handle, it is just a name in a global
       * namespace that any process can use.
       */
      using process_handle_t = pid_t;

      /**
       * The process ID of the child.
       */
      process_handle_t pid;

      /**
       * The exit status.  This is a cached result of the last wait call.  Once
       * the process has exited, this is used rather than querying anything
       * dynamically.
       */
      ExitStatus status;

      /**
       * Wrapper around the POSIX wait call that correctly handles
       * interruption.  Like most POSIX calls, `::waitpid` can spuriously return
       * if a signal is raised.  The wrapper also updates the `status` field and
       * returns the cached result if the process has exited (at which point the
       * PID may have been recycled and so calls to `::waitpid` will return
       * unpredictable results.  This isolates all of the details of correctly
       * using the underlying system call.
       */
      ExitStatus waitpid(int options = 0)
      {
        if (status.has_exited)
        {
          return status;
        }
        pid_t ret;
        int s;
        bool retry = false;
        do
        {
          ret = ::waitpid(pid, &s, options);
          retry = (ret == -1) && (errno == EINTR);
        } while (retry);
        if (ret == -1)
        {
          err(1, "Waiting for child failed");
        }
        if (ret == pid)
        {
          status = {true, WEXITSTATUS(s)};
        }
        return status;
      }

    public:
      /**
       * Constructor.  Takes a callable object that is run in the context of
       * the child process.
       */
      template<typename T>
      ChildProcessVFork(T&& start)
      {
        pid = vfork();
        if (pid == 0)
        {
          // If we have a platform-specific mechanism for requesting a SIGKILL
          // when the parent exits, use it.  This allows us to automatically
          // clean up sandboxes if the parent terminates abnormally.
          //
          // On Linux, we can't use `PR_SET_PDEATHSIG` because that causes the
          // child to receive the signal when the *thread* that called `vfork`
          // exits, not when the *process* exits.  See:
          // https://bugzilla.kernel.org/show_bug.cgi?id=43300
#  ifdef __FreeBSD__
          int deathsig = SIGKILL;
          procctl(P_PID, 0, PROC_PDEATHSIG_CTL, &deathsig);
#  endif
          start();
          SANDBOX_INVARIANT(0, "start function must not return");
        }
      }

      /**
       * Returns the current exit status.  This does not block.  If the process
       * has not exited, the return value will have `has_exited` set to `false`.
       */
      ExitStatus exit_status()
      {
        return waitpid(WNOHANG);
      }

      /**
       * Blocking call, does not return until the child has exited.
       */
      ExitStatus wait_for_exit()
      {
        return waitpid();
      }

      /**
       * Forcibly kill the child.
       */
      void terminate()
      {
        kill(pid, SIGKILL);
      }
    };
  }
}
#endif
