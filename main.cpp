#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <x86intrin.h>

namespace incrementer {
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

		class test_runner {
		public:
			void run(std::uint64_t participants, bool responsible = false)
			{
				constexpr auto total_increments = 1ull << 24;
				const auto target = total_increments / participants;
				synchronize(
					participants,
					[this, target] { return spinlock.run(target); },
					responsible);

				synchronize(
					participants,
					[this, target] { return atomic.run(target); },
					responsible);

				synchronize(
					participants,
					[this, target] { return null.run(target); },
					responsible);
			}

		private:
			std::atomic_uint64_t ready {};
			std::atomic_uint64_t done {};
			std::atomic_bool start {};
			spinlock_test spinlock;
			atomic_test atomic;
			null_test null;

			template <typename test_task>
			void synchronize(std::uint64_t participants, test_task&& run_test, bool responsible)
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

				run_test();

				++done;
				if (responsible) {
					while (done < participants)
						_mm_pause();

					start = false;
					done = 0;
				}

				while (start)
					_mm_pause();
			}
		};
	}
}

int main()
{
	using namespace incrementer;

	const auto core_count = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(core_count - 1);

	test_runner runner {};
	for (auto& thread : threads)
		thread = std::thread {[&runner, core_count] { runner.run(core_count); }};

	runner.run(core_count, true);

	for (auto& thread : threads)
		thread.join();
}