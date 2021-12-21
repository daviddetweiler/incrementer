#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

namespace incremementer {
	namespace {
		struct alignas(64) cache_line {
			std::uint64_t a;
			std::uint64_t b;
			std::uint64_t c;
			std::uint64_t d;
		};

		static_assert(alignof(cache_line) == 64);
		static_assert(sizeof(cache_line) == 64);
	}
}

int main()
{
	const auto core_count = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(core_count - 1);

	for (auto& thread : threads)
		thread.join();
}