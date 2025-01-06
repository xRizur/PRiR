#include "ParallelSwarm.h"
#include "MathHelper.h"
#include "consts.h"

#include <iostream>
#include <cmath>
#include <iomanip>  // std::setprecision
using namespace std;

// Ustaw sobie na true, by włączyć debug (wypisywanie).
static const bool DEBUG = true;

ParallelSwarm::ParallelSwarm(int robots,
                             Antenna *antenna,
                             Function *function)
    : Swarm(robots, antenna, function)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Podział robotów: local_count = robots / size, ale jeśli się nie dzieli:
    int base_count = robots / size;
    int remainder  = robots % size;

    // Każdy proces dostaje base_count, a ostatnie remainder rozdzielamy
    // np. w stylu: rank < remainder -> base_count+1
    // Prościej: rank < remainder: local_count = base_count+1, w pętli
    // Ale najpopularniej: dajmy "resztę" ostatniemu procesowi:
    if (rank < size - 1) {
        local_count = base_count;
    } else {
        // ostatni proces (rank = size-1) dostaje resztę
        local_count = base_count + remainder;
    }

    // Teraz musimy ustalić local_start: 
    // np. pętla i sumowanie dla ranków < rank (zależy, czy chcesz block distribution).
    // Dla minimalizmu:
    //   - rank < size-1: local_start = rank*base_count
    //   - rank == size-1: local_start = (size-1)*base_count
    // a local_end = local_start + local_count
    if (rank < size - 1) {
        local_start = rank * base_count;
    } else {
        local_start = (size - 1) * base_count;
    }
    local_end = local_start + local_count;

    step = 0;

    allocate_memory();
    initialize_antennas();

    // Debug info o podziale robotów
    if (DEBUG) {
        cerr << "[Rank " << rank << "] robots=" << robots 
             << " size=" << size 
             << " base_count=" << base_count
             << " remainder=" << remainder
             << " local_start=" << local_start
             << " local_end=" << local_end
             << " local_count=" << local_count
             << endl;
    }
}
void ParallelSwarm::before_first_run() {
    // Rozesłanie pozycji robotów do wszystkich procesów
    broadcast_positions();
    
    // Możesz dodać dodatkowe inicjalizacje tutaj, jeśli są potrzebne
}

void ParallelSwarm::broadcast_positions() {
    // Tylko rank=0 ma zainicjalizowane pozycje
    if (rank == 0) {
        // Inicjalizacja pozycji robotów
        // Zakładam, że `initialize_swarm` została już wywołana w main.cpp dla rank=0
    }

    // Rozesłanie pozycji robotów do wszystkich procesów
    for(int r = 0; r < robots; r++) {
        MPI_Bcast(position[r], dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Debug: sprawdź kilka pozycji po rozesłaniu
    if (rank == 0 || rank == 1) { // Możesz dodać więcej rang do sprawdzenia
        std::cerr << "[Rank " << rank << "] after Bcast | position[0..4][0] = ";
        for(int r = 0; r < 5 && r < robots; r++) {
            std::cerr << position[r][0] << " ";
        }
        std::cerr << std::endl;
    }
}
void ParallelSwarm::run(int steps) {
    for(int s = 0; s < steps; s++){
        single_step();
    }
}

void ParallelSwarm::single_step() {
    step++;
    // if (DEBUG) {
    //     cerr << "[Rank " << rank << "] step=" << step 
    //          << " BEFORE Evaluate | function_value[0..4] = ";
    //     for (int i = 0; i < 5 && i < robots; i++){
    //         cerr << function_value[i] << " ";
    //     }
    //     cerr << endl;
    // }
    // cerr << "[Rank " << rank << "] step=" << step << " | pos[0,0]=" << position[0][0] << " local_end=" << local_end << endl;
    evaluate_function_parallel();

    // (2) Debug: pokaż fragment function_value lokalnie (przed Allgather)
    // if (DEBUG) {
    //     cerr << "[Rank " << rank << "] step=" << step 
    //          << " BEFORE Allgather After Evaluate | function_value[0..4] = ";
    //     for (int i = 0; i < 5 && i < robots; i++){
    //         cerr << function_value[i] << " ";
    //     }
    //     cerr << endl;
    // }

    int base_count = robots / size;
    int remainder  = robots % size;

    // Tablice do Allgatherv
    int *recvcounts = new int[size];
    int *displs     = new int[size];
    for(int i=0; i<size; i++){
        recvcounts[i] = base_count;
        if (i == size-1) {
            recvcounts[i] += remainder; // reszta do ostatniego
        }
    }
    // displs[0] = 0, displs[i] = displs[i-1] + recvcounts[i-1]
    displs[0] = 0;
    for(int i=1; i<size; i++){
        displs[i] = displs[i-1] + recvcounts[i-1];
    }

    // Teraz wywołujemy MPI_Allgatherv zamiast Allgather
    MPI_Allgatherv(&function_value[local_start],  // sendbuf
                   local_count,                   // sendcount
                   MPI_DOUBLE,
                   function_value,                // recvbuf (docelowa tablica)
                   recvcounts,                    // ile elementów od i-tego procesu
                   displs,                        // gdzie trafia i-ty proces
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);

    delete[] recvcounts;
    delete[] displs;

    
    // if (DEBUG) {
    //     cerr << "[Rank " << rank << "] step=" << step 
    //          << " AFTER  Allgather | function_value[0..4] = ";
    //     for (int i = 0; i < 5 && i < robots; i++){
    //         cerr << function_value[i] << " ";
    //     }
    //     cerr << endl;
    // }

    // (5) find_neighbours_and_remember_best() – globalne pętle 0..robots
    find_neighbours_and_remember_best();

    // (6) move() – też globalne pętle
    move();

    // (7) fit_antenna_range() – globalnie
    fit_antenna_range();
}

void ParallelSwarm::evaluate_function_parallel() {
    // Każdy proces liczy wartość funkcji tylko dla [local_start..local_end)
    for (int robot = local_start; robot < local_end; robot++) {
        function_value[robot] = function->value(position[robot]);
    }
}

// Nieużywana w single_step, ale możesz ją zostawić
void ParallelSwarm::evaluate_function() {
    for (int r = 0; r < robots; r++) {
        function_value[r] = function->value(position[r]);
    }
}

void ParallelSwarm::find_neighbours_and_remember_best() {
    for (int robot = 0; robot < robots; robot++) {
        best_id = robot;
        nearest_neighbours[robot] = 0;

        best_function_value  = function_value[robot];
        my_antenna_range_sq  = antenna_range_sq[robot];
        my_position          = position[robot];

        for (int other_robot = 0; other_robot < robot; other_robot++) {
            compare_with_other_robot(robot, other_robot);
        }
        for (int other_robot = robot + 1; other_robot < robots; other_robot++) {
            compare_with_other_robot(robot, other_robot);
        }

        neighbour_id[robot] = best_id;
    }
}

void ParallelSwarm::compare_with_other_robot(int robot, int other_robot) {
    if (MathHelper::distanceSQ(my_position, position[other_robot], dimensions)
        < my_antenna_range_sq)
    {
        nearest_neighbours[robot]++;
        if (best_function_value < function_value[other_robot]) {
            best_function_value = function_value[other_robot];
            best_id = other_robot;
        }
    }
}

void ParallelSwarm::move() {
    for (int robot = 0; robot < robots; robot++) {
        MathHelper::move(
            position[robot],
            position[neighbour_id[robot]],
            new_position[robot],
            dimensions,
            STEP_SIZE / sqrt(step)
        );
    }

    for (int robot = 0; robot < robots; robot++) {
        for (int d = 0; d < dimensions; d++) {
            position[robot][d] = new_position[robot][d];
        }
    }
}

void ParallelSwarm::fit_antenna_range() {
    for (int robot = 0; robot < robots; robot++) {
        double range = antenna->range(
            sqrt(antenna_range_sq[robot]),
            nearest_neighbours[robot]
        );
        antenna_range_sq[robot] = range * range;
    }
}

void ParallelSwarm::allocate_memory() {
    position      = new double*[robots];
    new_position  = new double*[robots];
    for (int i = 0; i < robots; i++) {
        position[i]     = new double[dimensions];
        new_position[i] = new double[dimensions];
    }

    neighbour_id       = new int[robots];
    nearest_neighbours = new int[robots];
    function_value     = new double[robots];
    antenna_range_sq   = new double[robots];
}

void ParallelSwarm::initialize_antennas() {
    double vSQ = antenna->initial_range();
    vSQ *= vSQ;
    for (int r = 0; r < robots; r++) {
        antenna_range_sq[r] = vSQ;
    }
}

void ParallelSwarm::set_position(int dimension, int robot, double val) {
    position[robot][dimension] = val;
}

double ParallelSwarm::get_position(int robot, int dimension) {
    return position[robot][dimension];
}
