// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#ifdef USE_KQUEUE_PROCDESC
#  include <err.h>
#  include <errno.h>
#  include <sys/event.h>
#  include <sys/procdesc.h>
#  include <sys/types.h>
#  include <sys/wait.h>
namespace sandbox
{
  namespace platform
  {
    /**
     * Implementation of the child process interface using the process
     * descriptor interfaces.
     */
    class ChildProcessPDFork
    {
      /**
       * With process descriptors, process handles are just another kind
       * of file handle.
       */
      using process_handle_t = handle_t;

      /**
       * The handle to a kqueue that is used to wait for events from the child
       * process.
       */
      Handle kq;

      /**
       * The process descriptor for the child process.
       */
      Handle proc;

      /**
       * The cached exit status for the process.
       */
      ExitStatus status;

      /**
       * Wrapper around the `kqueue` call that takes a configurable timeout but
       * handles all of the updates of `status`.
       */
      ExitStatus wait_for_exit(struct timespec* timeout)
      {
        if (status.has_exited)
        {
          return status;
        }
        struct kevent event;
        // Poll for process exit.
        int ret;
        do
        {
          ret = kevent(kq.fd, nullptr, 0, &event, 1, timeout);
        } while ((ret == -1) && (errno == EINTR));
        if (ret == -1)
        {
          err(1, "Waiting for child failed");
        }
        if (ret == 1)
        {
          status = {true, static_cast<int>(WEXITSTATUS(event.data))};
        }
        return status;
      }

    public:
      /**
       * Constructor.  Takes a callable that is invoked in the child context.
       *
       * Currently, this runs in a throw-away CoW copy of the current process
       * but at some point I will get around to adding a `pdvfork` call to
       * FreeBSD and then this will have more vfork-like semantics.
       */
      template<typename T>
      ChildProcessPDFork(T&& start)
      {
        int pd;
        // Fork the child process.  `pd` now contains the process descriptor
        // and `pid` is used to determine whether the child is executing.
        pid_t pid = pdfork(&pd, PD_CLOEXEC);
        proc = pd;
        if (pid == 0)
        {
          start();
          assert(0 && "Should not be reached");
          abort();
        }
        assert(proc.is_valid());
        // Set up the kqueue that we can use to monitor for this process exiting
        kq = kqueue();
        struct kevent event;
        EV_SET(&event, proc.fd, EVFILT_PROCDESC, EV_ADD, NOTE_EXIT, 0, nullptr);
        if (kevent(kq.fd, &event, 1, nullptr, 0, nullptr) == -1)
        {
          err(1, "Setting up kqueue");
        }
      };

      /**
       * Poll for process exit.
       */
      ExitStatus exit_status()
      {
        // Time out immediately, return current status
        timespec timeout = {0, 0};
        return wait_for_exit(&timeout);
      }

      /**
       * Block until the process has exited.
       */
      ExitStatus wait_for_exit()
      {
        return wait_for_exit(nullptr);
      }

      /**
       * Forcibly kill the child.
       */
      void terminate()
      {
        pdkill(proc.fd, SIGKILL);
      }
    };
  }
}
#endif
