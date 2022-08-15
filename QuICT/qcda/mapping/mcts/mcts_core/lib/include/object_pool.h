#pragma once
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "mcts_node.h"
class MCTSNodePool {
 public:
  MCTSNodePool() {}

  void resize(uint64_t capacity) {
    this->capacity = capacity;
    memory = malloc(capacity * sizeof(mcts::MCTSNode));
    next = static_cast<mcts::MCTSNode*>(memory);
    end = next + capacity;
    free1 = new mcts::MCTSNode*[capacity + 1];
    frees = free1;
  }

  void clear() {
    size = 0;
    mcts::MCTSNode* start = static_cast<mcts::MCTSNode*>(memory);
    mcts::MCTSNode* cur_node = next.load();
    while (cur_node > start) {
      cur_node = next.fetch_sub(1);
      cur_node->clear();
    }
    cur_node = next.load();
    while (cur_node < start) {
      cur_node = next.fetch_add(1);
      cur_node++;
    }
    assert(cur_node == start);
    frees = free1;
  }

  ~MCTSNodePool() {
    free(memory);
    delete[] free1;
  }

  mcts::MCTSNode* create() {
    size++;
    mcts::MCTSNode** free = frees.load();
    if (free > free1) {
      free = frees.fetch_sub(1);
      free--;
      if (free >= free1) {
        if (*free < memory) printf("WTF=%p\n", *free);
        assert(*free >= memory);
        assert(*free <= end);
        return *free;
      }
    }

    mcts::MCTSNode* next1 = next.fetch_add(1);
    assert(next1 <= end);
    new (next1) mcts::MCTSNode();
    return next1;
  }

  void destroy(mcts::MCTSNode* o) {
    size--;
    mcts::MCTSNode** free = frees.fetch_add(1);
    while (free < free1) {
      free = frees.fetch_add(1);
    }
    *free = o;
    o->clear();
  }

  uint64_t getSize() { return size; }

  uint64_t getCapacity() { return capacity; }

  mcts::MCTSNode* getMemory() { return static_cast<mcts::MCTSNode*>(memory); }

 private:
  void* memory;
  std::atomic<mcts::MCTSNode*> next;
  mcts::MCTSNode* end;

  std::atomic<mcts::MCTSNode**> frees;
  mcts::MCTSNode** free1;

  std::atomic<uint64_t> size{0};
  uint64_t capacity;
};