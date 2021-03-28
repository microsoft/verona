#pragma once

#if defined(__linux__)

#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/ip.h>
#  include <netinet/tcp.h>
#  include <sys/epoll.h>
#  include <sys/socket.h>
#include <arpa/inet.h>

#  define BACKLOG 8192
#  define MAX_EVENTS 128

namespace verona::rt
{
  class LinuxTCPSocket
  {
  private:
    static void make_nonblocking(int sock)
    {
      int flags;

      flags = fcntl(sock, F_GETFL, 0);
      assert(flags >= 0);
      flags = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
      assert(flags >= 0);
    }

  public:
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

    static int socket_read(int fd, char* buf, int len)
    {
      return recv(fd, buf, len, 0);
    }

    static int socket_write(int fd, char* buf, int len)
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

      if (listen(sock, BACKLOG))
      {
        perror("listen");
        return -1;
      }
      return sock;
    }

    static int client_dial(char *ip, uint16_t port)
    {
      struct sockaddr_in sa;
      int ret, sock;

      ret = inet_pton(AF_INET, ip, &(sa.sin_addr));
      if (ret < 1)
      {
        perror("inet_pton");
        return -1;
      }

      sa.sin_family = AF_INET;
      sa.sin_port = htons(port);

      sock = socket(AF_INET, SOCK_STREAM, 0);
      if (!sock)
      {
        perror("socket");
        return -1;
      }

      make_nonblocking(sock);

      if(connect(sock, (struct sockaddr *)&sa, sizeof(struct sockaddr)) == -1)
      {
        perror("connect()");
        exit(1);
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

    static int register_socket(int efd, int fd, int flags, long cookie)
    {
      int ret;
      struct epoll_event ev;
      UNUSED(flags);

      ev.events = EPOLLIN;
      ev.data.ptr = (void*)cookie;
      ret = epoll_ctl(efd, EPOLL_CTL_ADD, fd, &ev);
      assert(!ret);

      return 0;
    }

    static int check_network_io(int efd, void** ptrs)
    {
      struct epoll_event events[MAX_EVENTS];
      int nfds, i;

      nfds = epoll_wait(efd, events, MAX_EVENTS, 0);
      for (i = 0; i < nfds; i++)
        ptrs[i] = events[i].data.ptr;

      return nfds;
    }
  };
}
#endif
