// C++ under LD_PRELOAD=libshardheap_malloc.so: new/delete, containers and
// strings handed between threads.
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int main() {
    std::mutex m;
    std::vector<std::unique_ptr<std::string>> bag;
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; t++) {
        ts.emplace_back([&, t] {
            std::map<int, std::string> local;
            for (int i = 0; i < 20000; i++) {
                local[i % 500] = std::string(16 + i % 300, 'a' + t);
                auto s = std::make_unique<std::string>(local[i % 500]);
                std::lock_guard<std::mutex> g(m);
                bag.push_back(std::move(s));
                if (bag.size() > 1000) bag.erase(bag.begin(), bag.begin() + 500);
            }
        });
    }
    for (auto &t : ts) t.join();
    std::printf("smoke.cpp: ok (%zu left)\n", bag.size());
    return 0;
}
