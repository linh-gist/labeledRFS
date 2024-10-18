#include <unordered_set>
#include <cstring>

struct Hasher {
    size_t size;
    size_t operator()(char* buf) const {
        // https://github.com/yt-project/yt/blob/c1569367c6e3d8d0a02e10d0f3d0bd701d2e2114/yt/utilities/lib/fnv_hash.pyx
        size_t hash_val = 2166136261;
        for (int i = 0; i < size; ++i) {
                hash_val ^= buf[i];
                hash_val *= 16777619;
        }
        return hash_val;
    }
};
struct Comparer {
    size_t size;
    bool operator()(char* lhs, char* rhs) const {
        return (std::memcmp(lhs, rhs, size) == 0) ? true : false;
    }
};

struct ArraySet {
    std::unordered_set<char*, Hasher, Comparer> set;

    ArraySet (size_t size) : set(0, Hasher{size}, Comparer{size}) {}
    ArraySet () {}

    bool add(char* buf) {
        auto p = set.insert(buf);
        return p.second;
    }
};