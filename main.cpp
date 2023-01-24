#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <x86intrin.h>

#include <pthread.h>

#include "zipf.h"

namespace incrementer {
	namespace {
		constexpr auto total_increments = 1ull << 28;
		constexpr auto kebibyte = 1024ull;
		constexpr auto mib32 = kebibyte * kebibyte * 32ull;
		constexpr auto gb1 = kebibyte * kebibyte * kebibyte;

		std::uint64_t dataset_size {};
		std::uint64_t cacheline_count {};
		std::uint64_t index_count {};
		std::uint64_t keyrange {};

		// Mutual exclusion lock.
		struct spinlock {
			uint locked; // Is the lock held?
		};

		struct alignas(64) line {
			volatile std::uint64_t value;
			spinlock lock;
		};

		static inline uint xchg(volatile uint* addr, uint newval)
		{
			uint result;

			// The + in "+m" denotes a read-modify-write operand.
			asm volatile("lock; xchgl %0, %1" : "+m"(*addr), "=a"(result) : "1"(newval) : "cc");

			return result;
		}

		// Check whether this cpu is holding the lock.
		int holding(struct spinlock* lock) { return lock->locked; }

		// Acquire the lock.
		// Loops (spins) until the lock is acquired.
		// Holding a lock for a long time may cause
		// other CPUs to waste time spinning to acquire it.
		void acquire(struct spinlock* lk)
		{
			// The xchg is atomic.
			// It also serializes, so that reads after acquire are not
			// reordered before it.
			while (xchg(&lk->locked, 1) != 0)
				;
		}

		// Release the lock.
		void release(struct spinlock* lk)
		{
			// The xchg serializes, so that reads before release are
			// not reordered after it.  The 1996 PentiumPro manual (Volume 3,
			// 7.2) says reads can be carried out speculatively and in
			// any order, which implies we need to serialize here.
			// But the 2007 Intel 64 Architecture Memory Ordering White
			// Paper says that Intel 64 and IA-32 will not move a load
			// after a store. So lock->locked = 0 would work here.
			// The xchg being asm volatile ensures gcc emits it after
			// the above assignments (and after the critical section).
			xchg(&lk->locked, 0);
		}

		static inline uint64_t RDTSC_START(void)
		{
			unsigned cycles_low, cycles_high;

			asm volatile("CPUID\n\t"
						 "RDTSC\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");

			return ((uint64_t)cycles_high << 32) | cycles_low;
		}

		/**
		 * CITE:
		 * http://www.intel.com/content/www/us/en/embedded/training/ia-32-ia-64-benchmark-code-execution-paper.html
		 */
		static inline uint64_t RDTSCP(void)
		{
			unsigned cycles_low, cycles_high;

			asm volatile("RDTSCP\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 "CPUID\n\t"
						 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");

			return ((uint64_t)cycles_high << 32) | cycles_low;
		}

		class alignas(64) _spinlock {
		public:
			void __attribute__((noinline)) lock()
			{
				while (!is_taken.test_and_set(std::memory_order_acquire))
					_mm_pause();
			}

			void __attribute__((noinline)) unlock() { is_taken.clear(std::memory_order_release); }

		private:
			std::atomic_flag is_taken {ATOMIC_FLAG_INIT};
		};

		class spinlock_test {
		public:
			spinlock_test() : values(cacheline_count) {}

			void __attribute_noinline__ run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				for (auto i = 0ull; i < per_thread; ++i) {
					const auto index = indexes[i & (index_count - 1)];
					const auto lock_addr = &values[index].lock;
					acquire(lock_addr);
					asm volatile("incq (%0)" ::"r"(&values[index]));
					release(lock_addr);
				}
			}

		private:
			std::vector<line> values;
		};

		class atomic_test {
		public:
			atomic_test() : values(cacheline_count) {}

			void __attribute_noinline__ run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				for (auto i = 0ull; i < per_thread; ++i)
					asm volatile("lock addq $1, (%0)" ::"r"(&values[indexes[i & (index_count - 1)]]));
			}

		private:
			std::vector<line> values;
		};

		class null_test {
		public:
			null_test() : values(cacheline_count) {}

			void __attribute_noinline__ run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				for (auto i = 0ull; i < per_thread; ++i)
					asm volatile("movq $1331, (%0)" ::"r"(&values[indexes[i & (index_count - 1)]]));
			}

		private:
			std::vector<line> values; // clunky
		};

		class fidget_spinner_test {
		public:
			fidget_spinner_test() : locks(64) {}

			void __attribute_noinline__ run(std::uint64_t per_thread, unsigned int id)
			{
				auto& lock = locks[id];
				for (auto i = 0ull; i < per_thread; ++i) {
					acquire(&lock.lock);
					release(&lock.lock);
				}
			}

		private:
			std::vector<line> locks {};
		};

		struct timings {
			std::uint64_t spinlock;
			std::uint64_t atomic;
			std::uint64_t null;
		};

		timings operator+(const timings& a, const timings& b)
		{
			return {a.spinlock + b.spinlock, a.atomic + b.atomic, a.null + b.null};
		}

		class test_runner {
		public:
			test_runner(double skew, unsigned int participants) :
				skew {skew},
				participants {participants},
				ready {},
				done {},
				start {},
				spinlock {},
				fidget {},
				atomic {},
				null {}
			{
			}

			timings run(unsigned int id)
			{
				const auto responsible = id == 0;
				const auto target = total_increments / participants;
				std::vector<std::uint64_t> indexes(index_count);
				zipf_distribution dist {skew, keyrange - 1, id};
				std::mt19937_64 engine {std::random_device {}()};
				for (auto& n : indexes)
					n = dist();

				const auto sl_time = synchronize(
					participants,
					[this, target, &indexes] { return spinlock.run(target, indexes); },
					responsible);

				const auto at_time = synchronize(
					participants,
					[this, target, &indexes] { return atomic.run(target, indexes); },
					responsible);

				const auto nl_time = synchronize(
					participants,
					[this, target, id] { return fidget.run(target, id); },
					responsible);

				return {sl_time, at_time, nl_time};
			}

		private:
			const double skew;
			const unsigned int participants;
			std::atomic_uint64_t ready;
			std::atomic_uint64_t done;
			std::atomic_bool start;
			spinlock_test spinlock;
			fidget_spinner_test fidget;
			atomic_test atomic;
			null_test null;

			template <typename test_task>
			std::uint64_t synchronize(std::uint64_t participants, test_task&& run_test, bool responsible)
			{
				++ready;
				if (responsible) {
					while (ready < participants)
						_mm_pause();

					ready = 0;
					start = true;
				}

				while (!start)
					_mm_pause();

				const auto start_time = RDTSC_START();
				run_test();
				const auto stop_time = RDTSCP();

				++done;
				if (responsible) {
					while (done < participants)
						_mm_pause();

					start = false;
					done = 0;
				}

				while (start)
					_mm_pause();

				return stop_time - start_time;
			}
		};

		struct averages {
			double spinlock;
			double atomic;
			double null;
		};

		struct data {
			test_runner* runner;
			timings* time;
			unsigned int id;
		};

		averages run_test(double skew)
		{
			static const auto core_count = std::thread::hardware_concurrency();
			std::vector<pthread_t> threads(core_count - 1);
			std::vector<data> datums(core_count - 1);
			std::vector<timings> times(core_count);

			test_runner runner {skew, core_count};
			auto id = 1u;
			const auto tproc = [](void* data_ptr) -> void* {
				const auto data = (struct data*)data_ptr;
				*data->time = data->runner->run(data->id);
				return NULL;
			};

			for (auto& thread : threads) {
				datums[id - 1] = {&runner, &times[id], id};
				pthread_attr_t attr;
				pthread_attr_init(&attr);

				cpu_set_t mask;
				CPU_ZERO(&mask);
				CPU_SET(id, &mask);
				pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);

				if (pthread_create(&thread, NULL, tproc, &datums[id - 1]))
					std::abort();

				++id;
			}

			times.front() = runner.run(0);

			for (const auto& thread : threads) {
				if (pthread_join(thread, NULL))
					std::abort();
			}

			const auto totals = std::accumulate(times.begin(), times.end(), timings {});
			const averages avg
				= {static_cast<double>(totals.spinlock) / total_increments,
				   static_cast<double>(totals.atomic) / total_increments,
				   static_cast<double>(totals.null) / total_increments};

			return avg;
		}
	}
}

int main(int argc, char** argv)
{
	using namespace incrementer;

	const auto large_arena = argc == 2 && std::strcmp(argv[1], "big") == 0;
	dataset_size = large_arena ? gb1 : mib32;
	cacheline_count = dataset_size / 64;
	index_count = cacheline_count;
	keyrange = cacheline_count; // 64ull * (1ull << 26);

	auto first = true;
	std::cout << "{\n";
	constexpr std::array skews {0.2,  0.4,	0.6,  0.8,	0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
								0.88, 0.89, 0.9,  0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
								0.99, 1.0,	1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09};

	for (auto skew : skews) {
		if (!first)
			std::cout << "," << std::endl;

		const auto avg = run_test(skew);
		std::cout << "\t" << skew << ": "
				  << "[" << avg.spinlock << ", " << avg.atomic << ", " << avg.null << "]";

		first = false;
	}

	std::cout << "}\n";
}