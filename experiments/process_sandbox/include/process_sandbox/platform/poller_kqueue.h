// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#if __has_include(<sys/event.h>)
#  include <sys/event.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * KQueue-based implementation of the Poller interface.  KQueue is
     * supported on most BSD-like systems natively and via a compatibility
     * library on Linux, Solaris, and Windows.
     */
    class KQueuePoller
    {
      /**
       * The kqueue used to monitor for events.  Multiple threads can safely
       * update a kqueue concurrently, so we don't need any other
       * synchronisation in this page.
       */
      int kq = kqueue();

    public:
      ~KQueuePoller()
      {
        close(kq);
      }

      /**
       * Add a new file descriptor that we'll wait for.  This can be called from
       * any thread without synchronisation.
       */
      void add(handle_t fd)
      {
        struct kevent event;
        EV_SET(&event, fd, EVFILT_READ, EV_ADD, 0, 0, nullptr);
        if (kevent(kq, &event, 1, nullptr, 0, nullptr) == -1)
        {
          snmalloc::DefaultPal::error("Setting up kqueue");
        }
      }

      /**
       * Wait for one of the registered file descriptors to become readable.
       * This blocks and returns true if there is a message, false if an error
       * occurred.  On success, `fd` will be set to the file descriptor
       * associated with the event and `eof` will be set to true if the socket
       * has been closed at the remote end, false otherwise.
       *
       * This may be called only from a single thread.
       */
      bool poll(handle_t& fd, bool& eof)
      {
        struct kevent event;
        // Wait for a single event.  We could wait for more and cache other
        // pending ones across multiple calls to this function as we do in the
        // `poll`-based implementation of this interface.  In current use we're
        // going to do several system calls of work between calls to this
        // function and so we shouldn't until it becomes a bottleneck.
        if (kevent(kq, nullptr, 0, &event, 1, nullptr) == -1)
        {
          return false;
        }
        fd = static_cast<handle_t>(event.ident);
        eof = (event.flags & EV_EOF) == EV_EOF;
        return true;
      }
    };
  }
}
#endif
