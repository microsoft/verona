#pragma once

#if defined(__linux__)

#  include "../ds/mpscq.h"

#  include <cassert>
#  include <cstdio>
#  include <fcntl.h>
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

    static int socket_config(int fd)
    {
      int one = 1;

      make_nonblocking(fd);
      if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (void*)&one, sizeof(one)))
      {
        perror("setsockopt(TCP_NODELAY)");
        return -1;
      }

      return 0;
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

    static int server_listen(uint16_t port)
    {
      int one, sock;
      struct sockaddr_in sin;

      sin.sin_family = AF_INET;
      sin.sin_addr.s_addr = htonl(0);
      sin.sin_port = htons(port);

      sock = socket(AF_INET, SOCK_STREAM, 0);
      if (!sock)
      {
        perror("socket");
        return -1;
      }

      make_nonblocking(sock);

      one = 1;
      if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, (void*)&one, sizeof(one)))
      {
        perror("setsockopt(SO_REUSEPORT)");
        return -1;
      }

      one = 1;
      if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (void*)&one, sizeof(one)))
      {
        perror("setsockopt(SO_REUSEADDR)");
        return -1;
      }

      if (bind(sock, (struct sockaddr*)&sin, sizeof(sin)))
      {
        perror("bind");
        return -1;
      }

      if (listen(sock, backlog))
      {
        perror("listen");
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
        perror("epoll_ctl(EPOLL_CTL_MOD)");
        assert(false);
      }
    }

    void epoll_event_delete(int fd)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_DEL, fd, nullptr);
      if (ret != 0)
      {
        perror("epoll_ctl(EPOLL_CTL_DEL)");
        assert(false);
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
