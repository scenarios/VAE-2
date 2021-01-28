class DLinkedNode():

    def __init__(self):
        self.key = None
        self.val = None
        self.pre = None
        self.nxt = None


class LRUCache:

    def __init__(self, capacity: int):

        self._capacity = capacity
        self._buffer = {}
        self._size = 0
        self._queue = DLinkedNode()

    def _add_to_head(self, node):

        if self._queue.nxt:
            node.pre, node.nxt = self._queue, self._queue.nxt
            self._queue.nxt = node
            node.nxt.pre = node
        else:
            node.pre, node.nxt = self._queue, self._queue.nxt
            self._queue.nxt = node
            self._queue_tail = node

    def _move_to_head(self, node):

        if node is self._queue_tail:
            node.pre.nxt = node.nxt
            self._queue_tail = self._queue_tail.pre
        else:
            node.pre.nxt = node.nxt
            node.nxt.pre = node.pre
        self._add_to_head(node)

    def _pop_tail(self) -> DLinkedNode:

        self._queue_tail = self._queue_tail.pre
        tail_node = self._queue_tail.nxt
        self._queue_tail.nxt = None

        return tail_node

    def get(self, key: int) -> int:

        knode = self._buffer.get(key, None)
        if knode:
            self._move_to_head(knode)
            return knode.val
        else:
            return -1

    def put(self, key: int, value: int) -> None:

        knode = self._buffer.get(key, None)
        if knode:
            self._move_to_head(knode)
        else:
            nnode = DLinkedNode()
            nnode.key, nnode.val = key, value
            if self._size < self._capacity:
                self._add_to_head(nnode)
                self._size += 1
            else:
                dnode = self._pop_tail()
                del self._buffer[dnode.key]
                self._add_to_head(nnode)
            self._buffer[key] = nnode


if __name__ == '__main__':

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
    cache = LRUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    cache.put(4, 4)
    cache.get(4)
    cache.get(3)
    cache.get(2)
    cache.get(1)

