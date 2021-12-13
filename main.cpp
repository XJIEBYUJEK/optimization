#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

using namespace std;

void print_results(double result, long num_ns) {
    cout << " Результат: " << result << '\n'
         << " Время: " << num_ns << " нс \n";
}

void integrate_default(double a, double b, unsigned n) {
    double dx = (b - a) / n;
    double result = 0;
    chrono::time_point<chrono::system_clock, chrono::nanoseconds> time_start, time_end;
    time_start = chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < n; ++i){
        const double x = a + dx * (double (i) + 0.5);
        result += x * x  * dx;
    }
    time_end = chrono::high_resolution_clock::now();

    print_results(result, (time_end - time_start) / 1ns);
}

void integrate_simd(double a, double b, unsigned n) {
    double dx = (b - a) / n;
    double result = 0;
    chrono::time_point<chrono::system_clock, chrono::nanoseconds> time_start, time_end;
    time_start = chrono::high_resolution_clock::now();
#pragma omp simd reduction(+ : result)
    for (unsigned i = 0; i < n; ++i){
        const double x = a + dx * (double (i) + 0.5);
        result += x * x  * dx;
    }
    time_end = chrono::high_resolution_clock::now();

    print_results(result, (time_end - time_start) / 1ns);
}

void integrate_parallel(double a, double b, unsigned n, int threads) {
    double dx = (b - a) / n;
    double result = 0;
    omp_set_num_threads(threads);
    chrono::time_point<chrono::system_clock, chrono::nanoseconds> time_start, time_end;
    time_start = chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : result) default(none) shared(a, n, dx)
    for (unsigned i = 0; i < n; ++i){
        const double x = a + dx * (double (i) + 0.5);
        result += x * x  * dx;
    }
    time_end = chrono::high_resolution_clock::now();

    print_results(result, (time_end - time_start) / 1ns);
}

int main(int argc, char **argv) {
    double a = stod(argv[1]);
    double b = stod(argv[2]);
    unsigned n = stoi(argv[3]);

    cout << "Default:\n";
    integrate_default(a, b, n);
    cout << "SIMD:\n";
    integrate_simd(a, b, n);
    cout << "PARALLEL:\n";
    for (int threads = 1; threads <= 4; ++threads) {
        cout << "Потоков: " << threads << "\n";
        integrate_parallel(a, b, n, threads);
    }

    return EXIT_SUCCESS;
}
