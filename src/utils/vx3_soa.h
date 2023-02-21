#ifndef VX3_SOA_H
#define VX3_SOA_H

#include "refl.hpp"
#include "utils/vx3_conf.h"
#include "utils/vx3_cuda.cuh"
#include <assert.h>
#include <cuda/std/tuple>
#include <vector>

#ifndef DEBUG_SOA
#define DEBUG_INPUT
#define DEBUG_OUTPUT_AND_ASSERT(index, max_size) assert(index < max_size);
#define DEBUG_FORWARD
#else
#define DEBUG_INPUT , const char *caller
#define DEBUG_OUTPUT_AND_ASSERT(index, max_size)                                         \
    if ((index) >= (max_size)) {                                                         \
        printf("Boundary exceeded, Caller: %s, index: %lu\n", caller, (index));          \
    }                                                                                    \
    assert((index) < (max_size));
#define DEBUG_FORWARD , ""
#endif
/**
 * The type the member returns when invoked.
 * In case of future updates when there exists function descriptors
 * among all descriptors, use the first one. The second one is just used to make
 * CLion IDE happy.
 */
// template<typename Member>
// using underlying_type = refl::trait::remove_qualifiers_t<decltype(
// Member()(std::declval<const typename Member::declaring_type &>()))>;

template <typename Member>
using custom_underlying_type =
    refl::trait::remove_qualifiers_t<typename Member::value_type>;

/**
 * Member lists of fields in SoA
 */

template <typename T>
constexpr auto filter_field_members = filter(refl::member_list<T>{}, [](auto member) {
    return is_field(member);
});

template <typename T>
using field_member_list = std::remove_const_t<decltype(filter_field_members<T>)>;

template <typename T>
constexpr auto
    filter_unreadable_field_members = filter(refl::member_list<T>{}, [](auto member) {
        return is_field(member) and not is_readable(member);
    });

template <typename T>
using unreadable_field_member_list = decltype(filter_unreadable_field_members<T>);

struct helper {
    // Provide unified access to regular fields and nested fields
    // For regular fields, T is a pointer type
    // For nested fields, T is VX3_hdStructOfArrays<SomeNestedField>
    template <typename T> static std::enable_if_t<std::is_class_v<T>> init(T &t) {}

    template <typename T> static std::enable_if_t<not std::is_class_v<T>> init(T &t) {
        t = nullptr;
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>> resize(T &t, size_t new_size,
                                                       const cudaStream_t &stream) {
        if (new_size == 0)
            t.free(stream);
        else
            t.resize(new_size, stream);
    }

    template <typename T>
    static std::enable_if_t<not std::is_class_v<T>> resize(T &t, size_t new_size,
                                                           const cudaStream_t &stream) {
        using type = typename std::remove_pointer<T>::type;
        if (t != nullptr)
            VcudaFreeAsync(t, stream);
        if (new_size == 0)
            t = nullptr;
        else
            VcudaMallocAsync(&t, new_size * sizeof(type), stream);
    }

    template <typename T, typename T2>
    static std::enable_if_t<std::is_class_v<T>> fill(T &t, T2 *&values, size_t num,
                                                     const cudaStream_t &stream) {
        t.fill(values, num, stream);
    }

    template <typename T, typename T2>
    static std::enable_if_t<not std::is_class_v<T>> fill(T &t, T2 *&values, size_t num,
                                                         const cudaStream_t &stream) {
        // Because arrays are flattened, use base element type
        using element_type = typename std::remove_extent_t<std::remove_pointer_t<T>>;
        VcudaMemcpyAsync(t, values, num * sizeof(element_type), cudaMemcpyHostToDevice,
                         stream);
    }

    template <typename T, typename T2>
    static std::enable_if_t<std::is_class_v<T>> read(T &t, T2 *values, size_t num,
                                                     const cudaStream_t &stream) {
        // num are not used for now since we assume reading the whole soa
        t.read(values, stream);
    }

    template <typename T, typename T2>
    static std::enable_if_t<not std::is_class_v<T>> read(T &t, T2 *values, size_t num,
                                                         const cudaStream_t &stream) {
        // Because arrays are flattened, use base element type
        using element_type = typename std::remove_extent_t<std::remove_pointer_t<T>>;
        VcudaMemcpyAsync(values, t, num * sizeof(element_type), cudaMemcpyDeviceToHost,
                         stream);
    }

    template <typename T, typename T2>
    __host__ __device__ static std::enable_if_t<std::is_array_v<T>> assign(T &t,
                                                                           T2 value) {
        constexpr size_t length = sizeof(T) / sizeof(std::remove_extent_t<T>);
        for (size_t i = 0; i < length; i++)
            t[i] = value[i];
    }

    template <typename T, typename T2>
    __host__ __device__ static std::enable_if_t<not std::is_array_v<T>> assign(T &t,
                                                                               T2 value) {
        t = value;
    }
};

// template <typename T, typename = std::enable_if_t<std::is_pointer_v<T>>>
//__device__ auto &getSub(T &ptr, size_t index) {
//    return ptr[index];
//}

template <typename T, size_t N> __device__ auto &getSub(T &soa, size_t index) {
    return soa.get<N>(index DEBUG_FORWARD);
}

template <typename T, size_t N, size_t N1, size_t... Ns>
__device__ auto &getSub(T &soa, size_t index) {
    return getSub<std::remove_reference_t<decltype(cuda::std::get<N>(soa.storage()))>, N1,
                  Ns...>(cuda::std::get<N>(soa.storage()), index);
}

template <typename T> struct VX3_ArrayAccessor {
    T *start_address;
    size_t offset;
    size_t stride;
    size_t max_length;
    __device__ VX3_ArrayAccessor(T *start_address, size_t stride, size_t offset,
                                 size_t max_length)
        : start_address(start_address), stride(stride), offset(offset),
          max_length(max_length){};
    __device__ T &operator[](size_t index) {
        assert(index < max_length);
        return *(start_address + index * stride + offset);
    }
};

template <typename T> struct VX3_ConstArrayAccessor {
    const T *start_address;
    size_t offset;
    size_t stride;
    size_t max_length;
    __device__ VX3_ConstArrayAccessor(const T *start_address, size_t stride,
                                      size_t offset, size_t max_length)
        : start_address(start_address), stride(stride), offset(offset),
          max_length(max_length){};
    __device__ const T &operator[](size_t index) {
        assert(index < max_length);
        return *(start_address + index * stride + offset);
    }
};

/**
 *  Structure of arrays created on the host side and read by device
 * @tparam T The type of structure to be converted
 */
template <typename T>
struct VX3_hdStructOfArrays
    : refl::runtime::proxy<VX3_hdStructOfArrays<T>, std::remove_cv_t<T>> {
  public:
    using members = refl::member_list<T>;
    using field_members = field_member_list<std::remove_cv_t<T>>;
    using unreadable_field_members = unreadable_field_member_list<std::remove_cv_t<T>>;
    static_assert(field_members::size > 0, "Type has no fields!");
    static_assert(unreadable_field_members::size == 0, "Type has unreadable fields!");

    explicit VX3_hdStructOfArrays() : _storage_size(0) {
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            helper::init(cuda::std::get<i>(_storage));
        });
    }

    ~VX3_hdStructOfArrays() = default;

    void free(const cudaStream_t &stream) {
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            helper::resize(cuda::std::get<i>(_storage), 0, stream);
        });
        VcudaStreamSynchronize(stream);
        _storage_size = 0;
    }

    void resize(size_t new_size, const cudaStream_t &stream) {
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            helper::resize(cuda::std::get<i>(_storage), new_size, stream);
        });
        VcudaStreamSynchronize(stream);
        _storage_size = new_size;
    }

    void fill(const T &input, const cudaStream_t &stream) {
        fill(std::vector<T>(_storage_size, input), stream);
    }

    void fill(const std::vector<T> &input, const cudaStream_t &stream) {
        auto tmp = new T[input.size()];
        for (size_t i = 0; i < input.size(); i++)
            tmp[i] = input[i];
        fill(tmp, input.size(), stream);
    }

    void fill(T *input, size_t num, const cudaStream_t &stream) {
        if (num > _storage_size)
            resize(num, stream);
        for_each(field_members{}, [&](auto member, size_t index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = custom_underlying_type<decltype(member)>;
            if constexpr (not std::is_array_v<type>) {
                auto tmp = new type[num];
                for (size_t j = 0; j < num; j++)
                    tmp[j] = member(input[j]);
                helper::fill(cuda::std::get<i>(_storage), tmp, num, stream);

                // Make sure all host side data are transferred
                VcudaStreamSynchronize(stream);
                delete[] tmp;
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indicies
                constexpr size_t length =
                    sizeof(type) / sizeof(std::remove_extent_t<type>);
                auto tmp = new std::remove_extent_t<type>[num * length];
                for (size_t j = 0; j < length; j++) {
                    for (size_t k = 0; k < num; k++) {
                        tmp[j * num + k] = member(input[k])[j];
                    }
                }
                // Note: since we flattens the array, copy num * length elements
                helper::fill(cuda::std::get<i>(_storage), tmp, num * length, stream);

                // Make sure all host side data are transferred
                VcudaStreamSynchronize(stream);
                delete[] tmp;
            }
        });
    }

    void read(std::vector<T> &output, const cudaStream_t &stream) {
        T *tmp = new T[_storage_size];
        read(tmp, stream);
        for (size_t i = 0; i < _storage_size; i++)
            output.emplace_back(tmp[i]);
    }

    void read(T *output, const cudaStream_t &stream) {
        for_each(field_members{}, [&](auto member, size_t index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = custom_underlying_type<decltype(member)>;
            if constexpr (not std::is_array_v<type>) {
                auto tmp = new type[_storage_size];
                helper::read(cuda::std::get<i>(_storage), tmp, _storage_size, stream);
                // Make sure all device side data are transferred
                VcudaStreamSynchronize(stream);

                for (size_t j = 0; j < _storage_size; j++)
                    member(output[j]) = tmp[j];
            } else {
                // Un-flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indicies
                constexpr size_t length =
                    sizeof(type) / sizeof(std::remove_extent_t<type>);
                auto tmp = new std::remove_extent_t<type>[_storage_size * length];

                // Note: since we flattens the array, copy _storage_size * length elements
                helper::read(cuda::std::get<i>(_storage), tmp, _storage_size * length,
                             stream);
                // Make sure all device side data are transferred
                VcudaStreamSynchronize(stream);

                for (size_t j = 0; j < length; j++) {
                    for (size_t k = 0; k < _storage_size; k++) {
                        member(output[k])[j] = tmp[j * _storage_size + k];
                    }
                }
                delete[] tmp;
            }
        });
    }

    /**
     * G1
     * Access the reference of the Nth (non-structure, non-array) field of the index-th
     * element: Eg: soa.get<0>(100) = 0
     */
    template <
        size_t N,
        std::enable_if_t<
            not std::is_class_v<custom_underlying_type<
                refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>> and
                not std::is_array_v<custom_underlying_type<
                    refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>>,
            bool> = true>
    __device__ auto &get(size_t index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        assert(index < _storage_size);
        return cuda::std::get<N>(_storage)[index];
    }

    /**
     * G1.1
     * Access the Nth array field of the index-th element using an accessor:
     * Eg: soa.get<0>(100)[3] = 0
     */
    template <size_t N,
              std::enable_if_t<std::is_array_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ auto get(size_t index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        using type = custom_underlying_type<
            refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>;
        using element_type = typename std::remove_extent_t<type>;
        constexpr size_t length = sizeof(type) / sizeof(element_type);
        // Use base type pointer to access values
        return VX3_ArrayAccessor(
            reinterpret_cast<element_type *>(cuda::std::get<N>(_storage)), _storage_size,
            index, length);
    }

    /**
     * G2
     * Access the const reference of the Nth (non-structure) field of the index-th
     * element: Eg: soa.get<0>(100) = 0
     */
    template <
        size_t N,
        std::enable_if_t<
            not std::is_class_v<custom_underlying_type<
                refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>> and
                not std::is_array_v<custom_underlying_type<
                    refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>>,
            bool> = true>
    __device__ const auto &get(size_t index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        return cuda::std::get<N>(_storage)[index];
    }

    /**
     * G2.1
     * Access the Nth array field of the index-th element using a const accessor:
     * Eg: soa.get<0>(100)[3] = 0
     */
    template <size_t N,
              std::enable_if_t<std::is_array_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ auto get(size_t index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        using type = custom_underlying_type<
            refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>;
        using element_type = typename std::remove_extent_t<type>;
        constexpr size_t length = sizeof(type) / sizeof(element_type);
        // Use base type pointer to access values
        return VX3_ConstArrayAccessor(
            reinterpret_cast<element_type *>(cuda::std::get<N>(_storage)), _storage_size,
            index, length);
    }

    /**
     * G3
     * Access a copy of the Nth (structure) field of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.get<1>(100).x                     // x is the field of a copy of sub field sec.
     *
     * Note: the method calls get(index) internally, which access every single element of
     * the second level structure in an aligned manner and copy them back.
     *
     * Note: returned value is const to prevent unintentional modify, since this would
     * have no effect
     */

    template <size_t N,
              std::enable_if_t<std::is_class_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ const auto get(size_t index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        // last get is a recursive call of G5
        return cuda::std::move(cuda::std::get<N>(_storage).get(index DEBUG_FORWARD));
    }

    /**
     * G4
     * Access the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.get<1, 2>(100)                     // A reference to sec.x of the 100th
     * element.
     *
     */
    template <size_t N, size_t N1, size_t... Ns,
              std::enable_if_t<std::is_class_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ auto &get(size_t index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        return getSub<decltype(*this), N, N1, Ns...>(*this, index);
    }

    /**
     * G5
     * Access a copy of the index-th element
     * Eg:
     * soa.get(100)
     */
    __device__ T get(size_t index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        T t{};
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            // recursive call of G1.1, G2, G3
            helper::assign(member(t), get<i>(index DEBUG_FORWARD));
        });
        return cuda::std::move(t);
    }

    /**
     * S1
     * Set the value of the Nth (non-structure, non-array) field of the index-th element:
     * Eg: soa.set<0>(100, 0)
     * Note: You can also use soa.get<0>(100) = 0
     */
    template <
        size_t N, typename T2,
        std::enable_if_t<
            not std::is_class_v<custom_underlying_type<
                refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>> and
                not std::is_array_v<custom_underlying_type<
                    refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>>,
            bool> = true>
    __device__ void set(size_t index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        cuda::std::get<N>(_storage)[index] = value;
    }

    /**
     * S1.1
     * Set Nth array field of the index-th element using an accessor:
     * Eg: soa.set<0>(100, a-pointer-to-array or array)
     */
    template <size_t N, typename T2,
              std::enable_if_t<std::is_array_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ void set(size_t index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        using type = custom_underlying_type<
            refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>>;
        using element_type = typename std::remove_extent_t<type>;
        constexpr size_t length = sizeof(type) / sizeof(element_type);

        // Use base type pointer to access values
        auto acc = VX3_ArrayAccessor(
            reinterpret_cast<element_type *>(cuda::std::get<N>(_storage)), _storage_size,
            index, length);
        for (size_t i = 0; i < length; i++)
            acc[i] = value[i];
    }

    /**
     * No corresponding S2
     */

    /**
     * S3
     * Set the value of the Nth (structure) field of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.set<1>(100, {10, 10, 10})
     * Note: since the mirrored get function returns a copy, you cannot
     * use the get function to set values of structure fields.
     */

    template <size_t N, typename T2,
              std::enable_if_t<std::is_class_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ void set(size_t index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        // last set is a recursive call of S5
        cuda::std::get<N>(_storage).set(index, value DEBUG_FORWARD);
    }

    /**
     * S4
     * Set the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.set<1, 2>(100, 0)                     // Set sec.x of the 100th element.
     * Note: you can also use soa.get<1, 2>(100) = 0
     */
    template <size_t N, size_t N1, size_t... Ns, typename T2,
              std::enable_if_t<std::is_class_v<custom_underlying_type<refl::trait::get_t<
                                   N, typename VX3_hdStructOfArrays<T>::members>>>,
                               bool> = true>
    __device__ void set(size_t index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        getSub<decltype(*this), N, N1, Ns...>(*this, index) = value;
    }

    /**
     * S5
     * Set the value of the index-th element
     * Eg:
     * soa.set(100, {...})
     */
    __device__ void set(size_t index, const T &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            helper::assign(get<i>(index DEBUG_FORWARD), member(value));
        });
    }

    /**
     * Call the first function on the index-th element, synchronize fields after
     * calling the function.
     */
    template <auto Func, bool NoRead = false, bool NoWrite = false, typename... Args>
    __device__ decltype(auto) call(size_t index, Args &&... args DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, _storage_size)
        T t{};
        FuncReturnGuard<NoWrite> guard(*this, t, index);
        if constexpr (not NoRead) {
            t = get(index DEBUG_FORWARD);
        }
        // See Section "pointers to member functions" at
        // https://en.cppreference.com/w/cpp/language/pointer#Pointers_to_members
        return (t.*Func)(std::forward<Args>(args)...);
    }

    __device__ __host__ size_t size() const { return _storage_size; }

    __device__ __host__ auto &storage() { return _storage; }

    template <typename Member> auto &operator[](Member) {
        constexpr auto i = refl::trait::index_of_v<Member, field_members>;
        return cuda::std::get<i>(_storage);
    }

    template <typename Member, typename Self>
    static auto invoke_impl(Self &&self, size_t index) -> decltype(auto) {
        constexpr auto i = refl::trait::index_of_v<Member, field_members>;
        auto &&vec = cuda::std::get<i>(self.storage_);
        return vec.at(index);
    }

  private:
    friend struct helper;
    template <bool NoWrite> struct FuncReturnGuard {
        VX3_hdStructOfArrays<T> &soa;
        T &target;
        size_t index;

        __device__ FuncReturnGuard(VX3_hdStructOfArrays<T> &soa, T &target, size_t index)
            : soa(soa), target(target), index(index) {}

        __device__ ~FuncReturnGuard() {
            if constexpr (not NoWrite) {
                soa.set(index, target DEBUG_FORWARD);
            }
        }
    };

    template <typename FieldMember> struct make_storage {
        using type = typename std::conditional<
            std::is_class_v<custom_underlying_type<FieldMember>>,
            VX3_hdStructOfArrays<custom_underlying_type<FieldMember>>,
            custom_underlying_type<FieldMember> *>::type;
    };

    template <typename> struct as_cuda_tuple;

    template <template <typename...> typename T2, typename... Ts>
    struct as_cuda_tuple<T2<Ts...>> {
        using type = cuda::std::tuple<Ts...>;
    };

    template <typename T2> using as_tuple_t = typename as_cuda_tuple<T2>::type;

    // Number of stored virtual structures
    size_t _storage_size;

    // Storage is a std::tuple containing device side memory pointers to
    // individual arrays Each array contains a property of the virtual
    // "Structure"
    as_tuple_t<refl::trait::map_t<make_storage, field_members>> _storage;
};

template <size_t N, typename T>
auto &getArrayPointerOfProperty(VX3_hdStructOfArrays<T> &soa) {
    using member = refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>;
    return soa[member{}];
}

template <size_t N, typename T>
const auto &getArrayPointerOfProperty(const VX3_hdStructOfArrays<T> &soa) {
    using member = refl::trait::get_t<N, typename VX3_hdStructOfArrays<T>::members>;
    return soa[member{}];
}

#define Pr(struct_name, property)                                                        \
    refl_impl::metadata::type_info__<struct_name>::field##_##property##_index

#endif // VX3_SOA_H