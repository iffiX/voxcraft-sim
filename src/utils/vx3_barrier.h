// Adapted from https://stackoverflow.com/questions/38999911/
// what-is-the-best-way-to-realize-a-synchronization-barrier-between-threads

#ifndef VX3_BARRIER_H
#define VX3_BARRIER_H
#include <assert.h>
#include <condition_variable>


class VX3_Barrier {

  public:
    explicit VX3_Barrier(std::size_t nb_threads)
        : m_mutex(), m_condition(), m_nb_threads(nb_threads) {
        assert(0u != m_nb_threads);
    }

    VX3_Barrier(const VX3_Barrier &barrier) = delete;

    VX3_Barrier(VX3_Barrier &&barrier) = delete;

    ~VX3_Barrier() noexcept { assert(0u == m_nb_threads); }

    VX3_Barrier &operator=(const VX3_Barrier &barrier) = delete;

    VX3_Barrier &operator=(VX3_Barrier &&barrier) = delete;

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);

        assert(0u != m_nb_threads);

        if (0u == --m_nb_threads) {
            m_condition.notify_all();
        } else {
            m_condition.wait(lock, [this]() { return 0u == m_nb_threads; });
        }
    }

  private:
    std::mutex m_mutex;

    std::condition_variable m_condition;

    std::size_t m_nb_threads;
};

#endif // VX3_BARRIER_H
