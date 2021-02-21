#pragma once

#if defined(__linux__)

#  include <cassert>
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

  class LinuxPoller
  {
  public:
    static int create_poll_fd()
    {
      return epoll_create1(0);
    }

    static void register_socket(int efd, int fd, int flags, Cown* cown)
    {
      UNUSED(flags);

      struct epoll_event ev;
      ev.events = EPOLLIN | EPOLLRDHUP;
      ev.data.ptr = cown;
      int ret = epoll_ctl(efd, EPOLL_CTL_ADD, fd, &ev);
      assert(!ret);
      UNUSED(ret);
    }

    static void unregister_socket(int efd, int fd)
    {
      int ret = epoll_ctl(efd, EPOLL_CTL_DEL, fd, nullptr);
      if (ret == -1)
      {
        perror("epoll_ctl(EPOLL_CTL_DEL)");
        assert(false);
      }
    }

    static size_t poll(int efd, Cown** cowns)
    {
      struct epoll_event events[max_events];
      const auto count = (size_t)epoll_wait(efd, events, max_events, 0);
      if (count == -(size_t)1)
      {
        perror("epoll_wait");
        assert(false);
        return 0;
      }

      for (size_t i = 0; i < count; i++)
        cowns[i] = (Cown*)events[i].data.ptr;

      return count;
    }
  };
}
#endif
