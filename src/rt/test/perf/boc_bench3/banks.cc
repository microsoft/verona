#include <algorithm>
#include <iostream>
#include <thread>
#include <mutex>
#include <cassert>
#include <vector>

std::mutex *accounts;
std::mutex logger;
uint64_t tx_count = 0;
int account_no;
int it_count;

// busy wait for 1ms
void busy_loop()
{
  // eat -> busy wait
  int it_count = 1000 / 10;
  for (int j=0;j<it_count;j++)
  {
    std::chrono::microseconds usec(10);
    auto start = std::chrono::system_clock::now();
    auto end = start + usec;

    // spin
    while (std::chrono::system_clock::now() <= end);
  }
}

void thread_main()
{
  int a, b;

  for (int i=0;i<it_count;i++)
  {
    a = rand() % account_no;
    b = rand() % account_no;
    while (b == a)
      b = rand() % account_no;

    {
      std::lock(accounts[a], accounts[b]);
      std::lock_guard<std::mutex> lk1(accounts[a], std::adopt_lock);
      std::lock_guard<std::mutex> lk2(accounts[b], std::adopt_lock);

      busy_loop();

      std::lock_guard<std::mutex> lg_lk(logger);
      tx_count++;
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    fprintf(stderr, "Usage ./exec <thread count> <account count> <per thread it>\n");
    return -1;
  }

  int thread_count = atoi(argv[1]);
  account_no = atoi(argv[2]);
  it_count = atoi(argv[3]);

  assert(account_no < 1000000);

  accounts = new std::mutex[(unsigned long)account_no];

  std::vector<std::thread> workers;
  for (int i = 0; i < thread_count; i++)
  {
    workers.push_back(std::thread([=]() { thread_main(); }));
  }

  std::for_each(workers.begin(), workers.end(), [](std::thread &t)
      {
      t.join();
      });
}
