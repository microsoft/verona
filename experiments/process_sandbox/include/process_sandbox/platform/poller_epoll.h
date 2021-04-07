// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#if __has_include(<sys/epoll.h>)
#  include <sys/epoll.h>

namespace sandbox
{
  namespace platform
  {
    /**
     * EPoll-based implementation of the Poller interface.  `epoll` is
     * Linux's equivalent of `kqueue`.
     */
    class EPollPoller
    {
      /**
       * The epoll interface used to monitor for events.  Multiple threads can
       * safely update this, so we don't need any other synchronisation in this
       * page.
       */
      int ep = epoll_create1(EPOLL_CLOEXEC);

    public:
      ~EPollPoller()
      {
        close(ep);
      }

      /**
       * Add a new file descriptor that we'll wait for.  This can be called from
       * any thread without synchronisation.
       */
      void add(handle_t fd)
      {
        struct epoll_event event;
        event.events = EPOLLIN;
        event.data.fd = fd;
        if (epoll_ctl(ep, EPOLL_CTL_ADD, fd, &event) == -1)
        {
          snmalloc::Pal::error("Setting up epoll");
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
        struct epoll_event event;
        event.data.fd = -1;
        event.events = 0;
        // Wait for a single event.  We could wait for more and cache other
        // pending ones across multiple calls to this function as we do in the
        // `poll`-based implementation of this interface.  In current use we're
        // going to do several system calls of work between calls to this
        // function and so we shouldn't until it becomes a bottleneck.
        int ret;
        do
        {
          ret = epoll_wait(ep, &event, 1, -1);
        } while ((ret == -1) && (errno == EINTR));
        if (ret == -1)
        {
          return false;
        }
        if (ret == 0)
        {
          return poll(fd, eof);
        }
        fd = static_cast<handle_t>(event.data.fd);
        eof = (event.events & EPOLLHUP) == EPOLLHUP;
        // epoll is level triggered or edge triggered for everything and because
        // we want to be level triggered for readability, we end up also being
        // level triggered for EOF notifications.  We therefore need to
        // explicitly unregister when we've reached the EOF condition.
        if (eof)
        {
          struct epoll_event event;
          event.data.fd = fd;
          epoll_ctl(ep, EPOLL_CTL_DEL, fd, &event);
        }
        return true;
      }
    };
  }
}
#endif
