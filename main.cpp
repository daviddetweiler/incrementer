#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <x86intrin.h>

namespace incremementer {
	namespace {
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

		class alignas(64) spinlock {
		public:
			void lock()
			{
				while (!is_taken.test_and_set(std::memory_order_acquire))
					_mm_pause();
			}

			void unlock() { is_taken.clear(std::memory_order_release); }

		private:
			std::atomic_flag is_taken {ATOMIC_FLAG_INIT};
		};

		class spinlock_test {
		public:
			void run(std::uint64_t per_thread)
			{
				for (auto i = 0ull; i < per_thread; ++i) {
					const std::unique_lock guard {lock};
					++value;
				}
			}

		private:
			alignas(64) std::uint64_t value {};
			spinlock lock;
		};

		class atomic_test {
		public:
			void run(std::uint64_t per_thread)
			{
				for (auto i = 0ull; i < per_thread; ++i)
					value.fetch_add(1);
			}

		private:
			alignas(64) std::atomic_uint64_t value {};
		};

		class null_test {
		public:
			void run(std::uint64_t per_thread)
			{
				for (auto i = 0ull; i < per_thread; ++i)
					++value;
			}

		private:
			alignas(64) std::uint64_t value {};
		};
	}
}

int main()
{
	const auto core_count = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(core_count - 1);

	for (auto& thread : threads)
		thread.join();
}