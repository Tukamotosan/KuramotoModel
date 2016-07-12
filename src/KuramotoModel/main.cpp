#include <stdio.h>
#include <omp.h>
#include "Simulator.h"

int main()
{
#pragma omp parallel for
	for (int i = 0; i < 10; i++) {
		printf("fuckin!! world!!!(%d)\n", i);
	}

	Simulator simulator;

	for (int i = 0; i < 100; i++) {
		simulator.exec2();
	}

	return 0;
}