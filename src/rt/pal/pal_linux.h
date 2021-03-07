#pragma once

#if defined(__linux__)

#  include "../ds/mpscq.h"
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

namespace verona::rt::io
{
  static constexpr size_t backlog = 8192;
  static constexpr size_t max_events = 128;

  class LinuxTCPSocket
  {
  public:
    static void make_nonblocking(int sock)
    {
      int flags;
      flags = fcntl(sock, F_GETFL, 0);
      assert(flags >= 0);
      flags = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
      assert(flags >= 0);
    }

    static int server_accept(int fd)
    {
      return accept(fd, nullptr, nullptr);
    }

    static int socket_read(int fd, char* buf, size_t len)
    {
      return recv(fd, buf, len, 0);
    }

    static int socket_write(int fd, char* buf, size_t len)
    {
      return send(fd, buf, len, MSG_NOSIGNAL);
    }

    static int close(int fd)
    {
      return ::close(fd);
    }

    static int socket_listen(const char* host, uint16_t port)
    {
      struct sockaddr_in addr;
      get_address(&addr, host, port);

      int sock = open_socket();
      if (!sock)
      {
        perror("socket");
        return -1;
      }

      int res = bind(sock, (struct sockaddr*)&addr, sizeof(addr));
      if (res == -1)
      {
        perror("bind");
        return -1;
      }

      res = listen(sock, backlog);
      if (res == -1)
      {
        perror("listen");
        return -1;
      }

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
    inline static int
    get_address(struct sockaddr_in* addr, const char* host, uint16_t port)
    {
      memset(addr, 0, sizeof(*addr));
      addr->sin_family = AF_INET;
      addr->sin_port = htons(port);

      if ((host == nullptr) || (host[0] == '\0'))
        addr->sin_addr.s_addr = htonl(0);
      else
        inet_pton(AF_INET, host, &addr->sin_addr.s_addr);

      return 0;
    }

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
      {
        perror("setsockopt(SO_REUSEADDR)");
        return -1;
      }

      return sock;
    }
  };

  template<typename T>
  class LinuxPoller
  {
    struct Msg
    {
      std::atomic<Msg*> next = nullptr;
      struct epoll_event ev = {};
      int fd = 0;
      bool cancel = false;

      static constexpr size_t size()
      {
        return sizeof(Msg);
      }
    };

    MPSCQ<Msg> q;
    std::atomic<size_t> event_count = 0;
    int efd;

    static struct epoll_event socket_event(T* cown)
    {
      struct epoll_event ev;
      ev.events = EPOLLONESHOT | EPOLLIN | EPOLLRDHUP;
      ev.data.ptr = cown;
      return ev;
    }

    void epoll_event_modify(int fd, struct epoll_event* ev)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_MOD, fd, ev);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_MOD, " << fd << ") "
                           << strerrorname_np(errno) << std::endl;
      }
    }

    void epoll_event_delete(int fd)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_DEL, fd, nullptr);
      if (ret != 0)
      {
        Systematic::cout() << "error: epoll_ctl(EPOLL_CTL_DEL, " << fd << ") "
                           << strerrorname_np(errno) << std::endl;
      }
    }

    void send_msg(Alloc* alloc, int fd, struct epoll_event ev, bool cancel)
    {
      auto* msg =
        new (alloc->alloc<sizeof(Msg)>()) Msg{nullptr, ev, fd, cancel};
      q.enqueue(msg);
    }

    void handle_msgs(Alloc* alloc)
    {
      while (true)
      {
        auto* msg = q.dequeue(alloc);
        if (msg == nullptr)
          break;

        if (msg->cancel)
          epoll_event_delete(msg->fd);
        else
          epoll_event_modify(msg->fd, &msg->ev);
      }
    }

  public:
    LinuxPoller()
    {
      efd = epoll_create1(0);
      auto* stub = new (ThreadAlloc::get()->alloc<sizeof(Msg)>()) Msg();
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

    void socket_register(int fd, T* cown)
    {
      auto ev = socket_event(cown);
      int ret = epoll_ctl(efd, EPOLL_CTL_ADD, fd, &ev);
      if (ret != 0)
      {
        perror("epoll_ctl(EPOLL_CTL_ADD)");
        assert(false);
      }
    }

    inline void socket_rearm(Alloc* alloc, int fd, T* cown, bool local)
    {
      auto ev = socket_event(cown);
      if (local)
        epoll_event_modify(fd, &ev);
      else
        send_msg(alloc, fd, ev, false);
    }

    inline void socket_deregister(Alloc* alloc, int fd, bool local)
    {
      if (local)
        epoll_event_delete(fd);
      else
        send_msg(alloc, fd, {}, true);
    }

    size_t poll(Alloc* alloc, T** cowns)
    {
      handle_msgs(alloc);

      struct epoll_event events[max_events];
      const int count = epoll_wait(efd, events, max_events, 0);
      if (count == -1)
      {
        perror("epoll_wait");
        assert(false);
        return 0;
      }

      for (size_t i = 0; i < (size_t)count; i++)
        cowns[i] = (T*)events[i].data.ptr;

      return (size_t)count;
    }
  };
}
#endif
