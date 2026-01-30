#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
    used = 0;
    peak = 0;
    max_peak = 0;
    ptr = nullptr;

    // 'alignment' defaults to sizeof(uint64_t), because it is the length of
    // the longest data type currently supported by the DataType field of
    // the tensor
    alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
    if (this->ptr != nullptr) {
        runtime->dealloc(this->ptr);
    }
}

size_t Allocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);

    // =================================== 作业 ===================================
    // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    // =================================== 作业 ===================================
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
        if (it->second >= size) {
            size_t addr = it->first;
            size_t leftover = it->second - size;
            free_blocks.erase(it);
            if (leftover > 0) {
                free_blocks[addr + size] = leftover;
            }
            used += size;
            return addr;
        }
    }
    size_t addr = peak;
    peak += size;
    if (peak > max_peak) {
        max_peak = peak;
    }
    used += size;
    return addr;
}

void Allocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    // =================================== 作业 ===================================
    // TODO: 设计一个算法来回收内存
    // =================================== 作业 ===================================
    used -= size;
    free_blocks[addr] = size;
    auto it = free_blocks.find(addr);

    // Merge with next block
    auto next = std::next(it);
    if (next != free_blocks.end() && next->first == addr + size) {
        it->second += next->second;
        free_blocks.erase(next);
    }

    // Merge with prev block
    if (it != free_blocks.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            free_blocks.erase(it);
            it = prev;
        }
    }

    // Shrink peak if possible
    if (it->first + it->second == peak) {
        peak = it->first;
        free_blocks.erase(it);
    }
}

void *Allocator::getPtr() {
    if (this->ptr == nullptr) {
        this->ptr = runtime->alloc(this->max_peak);
        printf("Allocator really alloc: %p %lu bytes\n", this->ptr, max_peak);
    }
    return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
    std::cout << "Used memory: " << this->used
              << ", peak memory: " << this->peak
              << ", max peak memory: " << this->max_peak << std::endl;
}
} // namespace infini
