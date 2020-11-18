#pragma once

#if defined(__linux__)

#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/ip.h>
#  include <netinet/tcp.h>
#  include <sys/epoll.h>
#  include <sys/socket.h>

#  define BACKLOG 8192

namespace verona::rt
{
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
  };
}
#endif
