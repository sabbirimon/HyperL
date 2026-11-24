// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::int64_t jobs = 4000000;
    std::vector<std::int32_t> finish;
    finish.reserve(jobs);
    finish.push_back(1);
    std::int64_t checksum = 1;
    for (std::int64_t i = 1; i < jobs; ++i) {
        const auto left = i - 1 - ((i * 4051 % 65521) % i);
        const auto right = i - 1 - ((i * 2654435761 % 4093) % i);
        const auto end = std::max(finish[left], finish[right]) + 1 + i % 13;
        finish.push_back(static_cast<std::int32_t>(end));
        checksum += end;
    }
    std::cout << checksum << '\n';
}
