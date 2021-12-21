#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

namespace incremementer {
	namespace {
		class test_data {
		public:
		private:
			alignas(64) std::uint64_t value[4];
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