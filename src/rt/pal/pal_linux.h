#pragma once

#if defined(__linux__)

#  include "../ds/mpscq.h"
#  include "../ds/stack.h"
#  include "../test/systematic.h"

#  include <arpa/inet.h>
#  include <cassert>
#  include <cstdio>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/ip.h>
#  include <netinet/tcp.h>
#  include <sys/epoll.h>
#  include <sys/socket.h>

namespace verona::rt
{
  class Cown;
}

namespace verona::rt::io
{
  using namespace snmalloc;

  static constexpr size_t backlog = 8192;
  static constexpr size_t max_events = 128;

  class LinuxEvent
  {
    friend MPSCQ<LinuxEvent>;
    friend class LinuxPoller;
    friend class LinuxTCP;

    std::atomic<LinuxEvent*> next{nullptr};
    struct epoll_event ev;
    int fd;
    MPSCQ<LinuxEvent>* destination = nullptr;

    LinuxEvent(int fd_, Cown* cown, uint32_t flags) : fd(fd_)
    {
      memset(&ev, 0, sizeof(ev));
      ev.events = flags | EPOLLONESHOT;
      ev.data.ptr = cown;
    }

    static inline LinuxEvent*
    create(Alloc* alloc, int fd_, Cown* cown_, uint32_t flags_)
    {
      return new (alloc->alloc<sizeof(LinuxEvent)>())
        LinuxEvent(fd_, cown_, flags_);
    }

  public:
    static constexpr size_t size()
    {
      return sizeof(LinuxEvent);
    }

    Cown* cown()
    {
      return (Cown*)ev.data.ptr;
    }
  };

  class LinuxPoller
  {
  private:
    MPSCQ<LinuxEvent> q;
    std::atomic<size_t> event_count = 0;
    int efd;

    void epoll_event_modify(LinuxEvent* event)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_MOD, event->fd, &event->ev);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_MOD, " << event->fd
                           << ") " << strerrorname_np(errno) << std::endl;
        assert(false);
      }
    }

    void handle_events(Alloc* alloc)
    {
      while (true)
      {
        auto* event = q.dequeue(alloc);
        if (event == nullptr)
          break;

        epoll_event_modify(event);
      }
    }

  public:
    LinuxPoller()
    {
      efd = epoll_create1(0);
      auto* alloc = ThreadAlloc::get_noncachable();
      auto* stub =
        new (alloc->alloc<LinuxEvent::size()>()) LinuxEvent(0, nullptr, 0);
      q.init(stub);
    }

    ~LinuxPoller()
    {
      auto* stub = q.destroy();
      ThreadAlloc::get_noncachable()->dealloc<sizeof(*stub)>(stub);
    }

    inline size_t get_event_count()
    {
      return event_count.load(std::memory_order_seq_cst);
    }

    inline size_t add_event_source()
    {
      return event_count.fetch_add(1, std::memory_order_seq_cst);
    }

    inline size_t remove_event_source()
    {
      const auto prev = event_count.fetch_sub(1, std::memory_order_seq_cst);
      assert(prev > 0);
      return prev;
    }

    void register_event(LinuxEvent& event)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_ADD, event.fd, &event.ev);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_ADD) "
                           << strerrorname_np(errno) << std::endl;
        assert(false);
      }
    }

    inline void set_destination(LinuxEvent& event)
    {
      event.destination = &q;
    }

    void handle_blocking_io(Stack<LinuxEvent, Alloc>& stack)
    {
      while (!stack.empty())
      {
        auto* event = stack.pop();
        event->destination->enqueue(event);
      }
      assert(stack.empty());
    }

    size_t poll(Alloc* alloc, Cown** cowns)
    {
      handle_events(alloc);

      struct epoll_event events[max_events];
      const int count = epoll_wait(efd, events, max_events, 0);
      if (count == -1)
      {
        Systematic::cout() << "error: epoll_wait " << strerrorname_np(errno)
                           << std::endl;
        return 0;
      }

      for (size_t i = 0; i < (size_t)count; i++)
        cowns[i] = (Cown*)events[i].data.ptr;

      return (size_t)count;
    }
  };

  static void make_nonblocking(int sock)
  {
    int flags;
    flags = fcntl(sock, F_GETFL, 0);
    assert(flags >= 0);
    flags = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    assert(flags >= 0);
  }

  static const char* errno_str()
  {
    return strerrorname_np(errno);
  }

  static bool would_block(int err)
  {
    return (err == EWOULDBLOCK) || (err == EAGAIN);
  }

  static constexpr uint32_t socket_flags = EPOLLIN | EPOLLRDHUP;

  class LinuxTCP
  {
  public:
    static LinuxEvent* event(Alloc* alloc, int fd, Cown* cown)
    {
      return LinuxEvent::create(alloc, fd, cown, socket_flags);
    }

    static LinuxEvent event(int fd, Cown* cown)
    {
      return LinuxEvent(fd, cown, socket_flags);
    }

    // static LinuxEvent* accept(Alloc* alloc, Cown* cown, LinuxEvent* event)
    // {
    //   int ret = ::accept(event->fd, nullptr, nullptr);
    //   if (ret == -1)
    //   {
    //     if (!would_block(ret))
    //       Systematic::cout() << "error: accept " << errno_str() <<
    //       std::endl;

    //     return nullptr;
    //   }

    //   make_nonblocking(ret);
    //   return LinuxEvent::create(alloc, ret, cown, socket_flags);
    // }
    static int accept(int socket)
    {
      int ret = ::accept(socket, nullptr, nullptr);
      if (ret == -1)
      {
        if (!would_block(ret))
          Systematic::cout() << "error: accept " << errno_str() << std::endl;

        return -1;
      }

      make_nonblocking(ret);
      return ret;
    }

    static int read(int fd, char* buf, size_t len)
    {
      return ::recv(fd, buf, len, 0);
    }

    static int write(int fd, const char* buf, size_t len)
    {
      return ::send(fd, buf, len, MSG_NOSIGNAL);
    }

    static int close(int fd)
    {
      return ::close(fd);
    }

    static int socket_listen(const char* host, uint16_t port)
    {
      auto* info = get_address_info(host, port);
      if (info == nullptr)
        return -1;

      int sock = -1;
      struct addrinfo* addr = info;
      for (; addr != nullptr; addr = addr->ai_next)
      {
        sock = open_socket(addr);
        if (sock != -1)
          break;
      }
      if (sock == -1)
        return -1;

      int res = bind(sock, addr->ai_addr, addr->ai_addrlen);
      if (res == -1)
      {
        Systematic::cout() << "error: bind " << strerrorname_np(errno)
                           << std::endl;
        return -1;
      }

      res = listen(sock, backlog);
      if (res == -1)
      {
        Systematic::cout() << "error: listen " << strerrorname_np(errno)
                           << std::endl;
        return -1;
      }

      freeaddrinfo(info);
      return sock;
    }

    static int socket_connect(const char* host, uint16_t port)
    {
      auto* info = get_address_info(host, port);
      if (info == nullptr)
        return -1;

      /// TODO: Happy Eyeballs
      int sock = -1;
      for (auto* p = info; p != nullptr; p = p->ai_next)
      {
        sock = open_socket(p);
        if (sock == -1)
          continue;

        auto res = connect(sock, p->ai_addr, p->ai_addrlen);
        if ((res == 0) || (errno == EINPROGRESS))
          break;

        res = close(sock);
        assert(res == 0);

        sock = -1;
      }

      freeaddrinfo(info);
      return sock;
    }

  private:
    inline static struct addrinfo*
    get_address_info(const char* host, uint16_t port)
    {
      // TODO: map any to loopback
      struct addrinfo hints;
      memset(&hints, 0, sizeof(hints));
      hints.ai_flags = AI_ADDRCONFIG;
      hints.ai_family = AF_UNSPEC;
      hints.ai_socktype = SOCK_STREAM;
      hints.ai_protocol = IPPROTO_TCP;
      char port_str[16];
      snprintf(port_str, sizeof(port_str), "%u", port);
      if ((host != nullptr) && (host[0] == '\0'))
        host = nullptr;

      struct addrinfo* info;
      int res = getaddrinfo(host, port_str, &hints, &info);
      if (res != 0)
        return nullptr;

      return info;
    }

    inline static int open_socket(struct addrinfo* info = nullptr)
    {
      int domain = AF_INET;
      int type = SOCK_STREAM;
      int protocol = 0;
      if (info != nullptr)
      {
        domain = info->ai_family;
        type = info->ai_socktype;
        protocol = info->ai_protocol;
      }
      int sock = socket(domain, type | SOCK_NONBLOCK, protocol);
      if (sock == -1)
      {
        Systematic::cout() << "error: socket " << strerrorname_np(errno)
                           << std::endl;
        return -1;
      }

      int opt_val = 1;
      int res =
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
      if (res == -1)
      {
        Systematic::cout() << "error: setsockopt(SO_REUSEADDR) "
                           << strerrorname_np(errno) << std::endl;
        return -1;
      }

      return sock;
    }
  };
}
#endif
