#ifndef __ZIPF_H__
#define __ZIPF_H__

#include <cstdint>

#include "zipf_distribution.hpp"

namespace incrementer {
	class zipf_distribution {
	public:
		zipf_distribution(double skew, std::uint64_t maximum, unsigned int) : distribution {maximum, skew} {}
		std::uint64_t operator()() { return distribution.sample(); }

	private:
    zipf_distribution_apache distribution;
	};
} // namespace kmercounter

#endif
