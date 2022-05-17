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
		constexpr auto total_increments = 1ull << 28;
		constexpr auto cacheline_count = 1ull << 16;
		constexpr auto keyrange = cacheline_count; // 64ull * (1ull << 26);

		struct alignas(64) line {
			volatile std::uint64_t value;
		};

		extern "C" std::uint64_t run_null_test(std::uint64_t iterations, line* values, const std::uint64_t* indices);
		extern "C" std::uint64_t run_atomic_test(std::uint64_t iterations, line* values, const std::uint64_t* indices);
		
		extern "C" std::uint64_t
		run_lock_test(std::uint64_t iterations, line* values, const std::uint64_t* indices, line* locks);

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
			spinlock_test() : values(cacheline_count), locks(cacheline_count) {}

			void run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				run_lock_test(per_thread, values.data(), indexes.data(), locks.data());
			}

		private:
			std::vector<line> values;
			std::vector<line> locks;
		};

		class atomic_test {
		public:
			atomic_test() : values(cacheline_count) {}

			void run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				run_atomic_test(per_thread, values.data(), indexes.data());
			}

		private:
			std::vector<line> values;
		};

		class null_test {
		public:
			null_test() : values(cacheline_count) {}

			void run(std::uint64_t per_thread, std::vector<std::uint64_t>& indexes)
			{
				run_null_test(per_thread, values.data(), indexes.data());
			}

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

		class test_runner {
		public:
			test_runner(double skew, unsigned int participants) :
				skew {skew},
				participants {participants},
				ready {},
				done {},
				start {},
				spinlock {},
				atomic {},
				null {}
			{
			}

			timings run(unsigned int id)
			{
				const auto responsible = id == 0;
				const auto target = total_increments / participants;
				std::vector<std::uint64_t> indexes(target);
				zipf_distribution dist {skew, keyrange, id};
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
					[this, target, &indexes] { return null.run(target, indexes); },
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

		averages run_test(double skew)
		{
			static const auto core_count = std::thread::hardware_concurrency();
			std::vector<std::thread> threads(core_count - 1);
			std::vector<timings> times(core_count);

			test_runner runner {skew, core_count};
			auto id = 1u;
			for (auto& thread : threads)
				thread = std::thread {[&runner, &time = times[id++], id] { time = runner.run(id); }};

			times.front() = runner.run(0);

			for (auto& thread : threads)
				thread.join();

			const auto totals = std::accumulate(times.begin(), times.end(), timings {});
			const averages avg
				= {static_cast<double>(totals.spinlock) / total_increments,
				   static_cast<double>(totals.atomic) / total_increments,
				   static_cast<double>(totals.null) / total_increments};

			// std::cout << "Spinlock took " << totals.spinlock << " cycles\n";
			// std::cout << "Atomic took " << totals.atomic << " cycles\n";
			// std::cout << "Null took " << totals.null << " cycles\n";
			// std::cout << "Average spinlock: " << avg.spinlock << "\n";
			// std::cout << "Average atomic: " << avg.atomic << "\n";
			// std::cout << "Average null: " << avg.null << "\n";

			return avg;
		}
	}
}

int main()
{
	using namespace incrementer;

	auto first = true;
	std::cout << "{\n";
	for (auto i = 0.0; i < 0.99; i += 0.05) {
		if (!first)
			std::cout << "," << std::endl;

		const auto avg = run_test(i);
		std::cout << "\t" << i << ": "
				  << "[" << avg.spinlock << ", " << avg.atomic << ", " << avg.null << "]";

		first = false;
	}

	std::cout << "}\n";
}