// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <cpp/when.h>
#include <test/harness.h>

using namespace verona::cpp;

// Binary tree with weak references for parent pointers
class Tree
{
public:
  typename cown_ptr<Tree>::weak parent;
  cown_ptr<Tree> left;
  cown_ptr<Tree> right;

  ~Tree()
  {
    Logging::cout() << "Node destroyed" << Logging::endl;
  }
};

// Build a balanced binary tree of a given depth
void build(cown_ptr<Tree>& curr, cown_ptr<Tree>::weak parent, size_t depth)
{
  when(curr) << [parent = std::move(parent), depth](acquired_cown<Tree> curr) {
    curr->parent = std::move(parent);
    if (depth > 0)
    {
      curr->left = make_cown<Tree>();
      curr->right = make_cown<Tree>();
      build(curr->left, curr.cown().get_weak(), depth - 1);
      build(curr->right, curr.cown().get_weak(), depth - 1);
    }
  };
}

// Walk up tree using parent pointers where they haven't been collected.
void up(cown_ptr<Tree>& curr)
{
  when(curr) << [](acquired_cown<Tree> curr) {
    auto parent = curr->parent.promote();
    if (parent)
    {
      Logging::cout() << "Parent is alive" << Logging::endl;
      up(parent);
    }
  };
}

// Recursively decent the tree using strong references, and
// perform up for each node.
void down(cown_ptr<Tree>& curr)
{
  when(curr) << [](acquired_cown<Tree> curr) {
    Logging::cout() << "Node is alive" << Logging::endl;
    if (curr->left)
      down(curr->left);
    if (curr->right)
      down(curr->right);
  };
  up(curr);
}

void test_body()
{
  auto tree = make_cown<Tree>();

  build(tree, cown_ptr<Tree>::weak(), 3);

  down(tree);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test_body);

  return 0;
}
