#include "MemoryManager.h"
#include <iostream>

int main() {
	MemoryManager manager;
	size_t val = 1 << 32;
	manager.AllocateDevice(val);
}
