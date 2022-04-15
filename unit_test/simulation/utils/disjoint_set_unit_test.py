from QuICT.simulation.utils.disjoint_set import DisjointSet


def test_set():
    print()
    cnt = 5
    disjoint_set = DisjointSet(cnt)

    for i in range(cnt):
        print(disjoint_set.find(i))

    disjoint_set.union(1, 3)
    disjoint_set.union(2, 4)

    for i in range(cnt):
        print(disjoint_set.find(i))
