#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <x86intrin.h>

#include "zipf.h"

namespace incrementer {
	namespace {
		constexpr auto cacheline_count = 1ull << 16;

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
			struct alignas(64) line {
				std::uint64_t value;
			};

			spinlock_test() : values(cacheline_count) {}

			void run(std::uint64_t per_thread)
			{
				auto& value = values.front().value;
				for (auto i = 0ull; i < per_thread; ++i) {
					const std::unique_lock guard {lock};
					++value;
				}
			}

			void check(std::uint64_t n) { assert(n == value); }

		private:
			std::vector<line> values;
			spinlock lock;
		};

		class atomic_test {
		public:
			struct alignas(64) line {
				std::atomic_uint64_t value;
			};

			atomic_test() : values(cacheline_count) {}

			void run(std::uint64_t per_thread)
			{
				auto& value = values.front().value;
				for (auto i = 0ull; i < per_thread; ++i) {
					auto old = value.load(std::memory_order::memory_order_relaxed);
					while (value.compare_exchange_strong(old, old + 1, std::memory_order::memory_order_relaxed))
						_mm_pause();
				}
			}

			void check(std::uint64_t n) { assert(n == value); }

		private:
			std::vector<line> values;
		};

		class null_test {
		public:
			struct alignas(64) line {
				volatile std::uint64_t value;
			};

			null_test() : values(cacheline_count) {}

			void run(std::uint64_t per_thread)
			{
				auto& value = values.front().value;
				for (auto i = 0ull; i < per_thread; ++i)
					++value;
			}

			void check(std::uint64_t n) { assert(n == value); }

		private:
			std::vector<line> values; // clunky
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

		constexpr auto total_increments = 1ull << 24;

		class test_runner {
		public:
			timings run(std::uint64_t participants, bool responsible = false)
			{
				const auto target = total_increments / participants;
				const auto real_total = target * participants;
				const auto sl_time = synchronize(
					participants,
					[this, target] { return spinlock.run(target); },
					responsible);

				const auto at_time = synchronize(
					participants,
					[this, target] { return atomic.run(target); },
					responsible);

				const auto nl_time = synchronize(
					participants,
					[this, target] { return null.run(target); },
					responsible);

				spinlock.check(real_total);
				atomic.check(real_total);
				null.check(real_total);

				return {sl_time, at_time, nl_time};
			}

		private:
			std::atomic_uint64_t ready {};
			std::atomic_uint64_t done {};
			std::atomic_bool start {};
			spinlock_test spinlock;
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
	}
}

int main()
{
	using namespace incrementer;

	const auto core_count = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(core_count - 1);
	std::vector<timings> times(core_count);

	test_runner runner {};
	auto id = 1u;
	for (auto& thread : threads)
		thread = std::thread {[&runner, &time = times[id++], core_count] { time = runner.run(core_count); }};

	times.front() = runner.run(core_count, true);
	const auto totals = std::accumulate(times.begin(), times.end(), timings {});
	std::cout << "Spinlock took " << totals.spinlock << " cycles\n";
	std::cout << "Atomic took " << totals.atomic << " cycles\n";
	std::cout << "Null took " << totals.null << " cycles\n";
	std::cout << "Average spinlock: " << static_cast<double>(totals.spinlock) / total_increments << "\n";
	std::cout << "Average atomic: " << static_cast<double>(totals.atomic) / total_increments << "\n";
	std::cout << "Average null: " << static_cast<double>(totals.null) / total_increments << "\n";

	for (auto& thread : threads)
		thread.join();
}