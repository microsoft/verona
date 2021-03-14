// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
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

  class LinuxEvent
  {
    friend class LinuxPoller;
    friend class LinuxTCP;
    struct epoll_event ev;
    int fd;

    LinuxEvent(int fd_, Cown* cown, uint32_t flags) : fd(fd_)
    {
      assert(fd != -1);
      ev.events = flags | EPOLLONESHOT;
      ev.data.ptr = cown;
    }

  public:
    void set_cown(Cown* cown)
    {
      ev.data.ptr = cown;
    }

    Cown* cown()
    {
      return (Cown*)ev.data.ptr;
    }
  };

  class LinuxPoller
  {
  public:
    static constexpr size_t max_events = 128;

    class Msg
    {
    private:
      friend MPSCQ<Msg>;
      friend LinuxPoller;

      std::atomic<Msg*> next{nullptr};
      MPSCQ<Msg>* destination;
      LinuxEvent event;

      Msg(MPSCQ<Msg>* destination_, LinuxEvent event_)
      : destination(destination_), event(event_)
      {}

      static Msg*
      create(Alloc* alloc, MPSCQ<Msg>* destination_, LinuxEvent event_)
      {
        return new (alloc->alloc<sizeof(Msg)>()) Msg(destination_, event_);
      }

      static constexpr size_t size()
      {
        return sizeof(Msg);
      }
    };

  private:
    MPSCQ<Msg> q;
    std::atomic<size_t> event_count = 0;
    int efd;

    inline void epoll_event_modify(LinuxEvent& event)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_MOD, event.fd, &event.ev);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_MOD) "
                           << strerrorname_np(errno) << " (cown "
                           << event.cown() << ")" << std::endl;
        assert(false);
      }
    }

    inline void handle_msgs(Alloc* alloc)
    {
      while (true)
      {
        auto* msg = q.dequeue(alloc);
        if (msg == nullptr)
          break;

        epoll_event_modify(msg->event);
      }
    }

  public:
    LinuxPoller()
    {
      efd = epoll_create1(0);
      auto* alloc = ThreadAlloc::get_noncachable();
      q.init(Msg::create(alloc, nullptr, LinuxEvent(0, nullptr, 0)));
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

    inline void register_event(LinuxEvent& event)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_ADD, event.fd, &event.ev);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_ADD) "
                           << strerrorname_np(errno) << " (cown "
                           << event.cown() << ")" << std::endl;
        assert(false);
      }
    }

    inline Msg* create_msg(Alloc* alloc, LinuxEvent& event)
    {
      assert(event.cown() != nullptr);
      return new (alloc->alloc<Msg::size()>()) Msg(&q, event);
    }

    inline void handle_blocking_io(Stack<Msg, Alloc>& stack)
    {
      while (!stack.empty())
      {
        auto* msg = stack.pop();
        msg->destination->enqueue(msg);
      }
      assert(stack.empty());
    }

    size_t poll(Alloc* alloc, Cown** cowns)
    {
      handle_msgs(alloc);

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

  static inline bool error_would_block(int err)
  {
    return (err == EWOULDBLOCK) || (err == EAGAIN);
  }

  template<typename T>
  class LinuxResult
  {
    friend class LinuxTCP;

    std::optional<T> value = {};
    int err = 0;

  public:
    LinuxResult(T v) : value(v) {}
    LinuxResult(int e) : err(e) {}

    T operator*()
    {
      return *value;
    }

    bool ok() const
    {
      return value.has_value();
    }

    const char* error() const
    {
      return strerrorname_np(err);
    }

    bool would_block() const
    {
      return error_would_block(err);
    }

    template<typename U>
    LinuxResult<U> forward_err()
    {
      assert(!ok());
      return LinuxResult<U>(err);
    }
  };

  class LinuxTCP
  {
    static constexpr size_t backlog = 8192;
    static constexpr uint32_t socket_flags = EPOLLIN | EPOLLRDHUP;

  public:
    static LinuxResult<LinuxEvent> connect(const char* host, uint16_t port)
    {
      auto* info = get_address_info(host, port);
      if (info == nullptr)
        return errno;

      // TODO: Happy Eyeballs
      int sock = -1;
      for (auto* p = info; p != nullptr; p = p->ai_next)
      {
        sock = open_socket(p);
        if (sock == -1)
          continue;

        auto res = ::connect(sock, p->ai_addr, p->ai_addrlen);
        if ((res == 0) || (errno == EINPROGRESS))
          break;

        res = ::close(sock);
        assert(res == 0);

        sock = -1;
      }

      freeaddrinfo(info);
      return LinuxEvent(sock, nullptr, socket_flags);
    }

    static LinuxResult<LinuxEvent> listen(const char* host, uint16_t port)
    {
      auto* info = get_address_info(host, port);
      if (info == nullptr)
        return errno;

      int sock = -1;
      struct addrinfo* addr = info;
      for (; addr != nullptr; addr = addr->ai_next)
      {
        sock = open_socket(addr);
        if (sock != -1)
          break;
      }
      if (sock == -1)
        return errno;

      int res = bind(sock, addr->ai_addr, addr->ai_addrlen);
      if (res == -1)
        return errno;

      res = ::listen(sock, backlog);
      if (res == -1)
        return errno;

      freeaddrinfo(info);
      return LinuxEvent(sock, nullptr, socket_flags);
    }

    static LinuxResult<LinuxEvent> accept(LinuxEvent& listener, Cown* cown)
    {
      int ret = ::accept(listener.fd, nullptr, nullptr);
      if (ret == -1)
        return errno;

      make_nonblocking(ret);
      return LinuxEvent(ret, cown, socket_flags);
    }

    static LinuxResult<size_t> read(LinuxEvent& event, char* buf, size_t len)
    {
      auto res = ::recv(event.fd, buf, len, 0);
      if (res == -1)
        return errno;

      return (size_t)res;
    }

    static LinuxResult<size_t>
    write(LinuxEvent& event, const char* buf, size_t len)
    {
      auto res = ::send(event.fd, buf, len, MSG_NOSIGNAL);
      if (res == -1)
        return errno;

      return (size_t)res;
    }

    static LinuxResult<bool> close(LinuxEvent& event)
    {
      auto res = ::close(event.fd);
      if (res == -1)
        return errno;

      return true;
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
        return -1;

      int opt_val = 1;
      int res =
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
      if (res == -1)
        return -1;

      return sock;
    }
  };
}
#endif
