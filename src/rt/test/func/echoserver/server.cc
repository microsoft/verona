#include <iostream>
#include <verona.h>

using namespace verona::rt;

struct serve_b : public VBehaviour<serve_b>
{
  TCPSock* sock;

  serve_b(TCPSock* sock_) : sock(sock_) {}
  void f()
  {
    char buf[64];
    int ret;

    ret = sock->socket_read(buf, 64);
    if (ret > 0)
      sock->socket_write(buf, ret);

    Cown::schedule<serve_b>(sock, sock);
  }
};

struct accept_b : public VBehaviour<accept_b>
{
  TCPSock* sock;

  accept_b(TCPSock* sock_) : sock(sock_) {}
  void f()
  {
    auto new_sock = sock->server_accept();
    if (new_sock)
    {
      std::cout << "Received new conn" << std::endl;
      Cown::schedule<serve_b>(new_sock, new_sock);
    }

    Cown::schedule<accept_b>(sock, sock);
  }
};

void open_server_conn(int port)
{
  auto server_conn = TCPSock::server_listen(port);
  Cown::schedule<accept_b>(server_conn, server_conn);
}

int main(int argc, char** argv)
{
  int nr_cpu, port;

  if (argc < 3)
  {
    fprintf(stderr, "Usage: ./server <thread_count> <port>\n");
    return -1;
  }

  nr_cpu = atoi(argv[1]);
  port = atoi(argv[2]);

  auto& sched = Scheduler::get();
  Scheduler::set_detect_leaks(true);
  sched.set_fair(true);
  sched.init(nr_cpu);

  sched.run_with_startup(&open_server_conn, port);
}
