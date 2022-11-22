#ifndef VX3_HASH_H
#define VX3_HASH_H

#include <functional>

template <typename T> struct UnorderedPair {
    T value1, value2;
    UnorderedPair(T value1, T value2) : value1(value1), value2(value2) {};
    bool operator==(const UnorderedPair &other) const {
        return (value1 == other.value1 && value2 == other.value2) ||
               (value2 == other.value1 && value1 == other.value2);
    };
};

template <typename T> struct UnorderedPairHash {
    std::size_t operator()(const UnorderedPair<T> &pair) const {
        return std::hash<T>()(pair.value1) ^ std::hash<T>()(pair.value2);
    }
};

#endif