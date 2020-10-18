#include "test_geodesics_ptp.h"

#include <cassert>
#include "che_off.h"
#include "geodesics_ptp.h"

using namespace std;


// geometry processing and shape analysis framework
namespace gproshan {


void main_test_geodesics_ptp(const int& nargs, const char** args)
{
    if (nargs < 2) {
        printf(
            "./test_geodesics [data_path]"
            "[n_test = 10]\n");
        return;
    }

    string filename = args[1];

    int n_test = nargs == 5 ? atoi(args[4]) : 10;

    vector<index_t> source = {0};

    che*   mesh = new che_off(filename);
    size_t n_vertices = mesh->n_vertices();

    index_t*        toplesets = new index_t[n_vertices];
    index_t*        sorted_index = new index_t[n_vertices];
    vector<index_t> limits;
    mesh->compute_toplesets(toplesets, sorted_index, limits, source);


    double ptp_cpu_time =
        test_ptp_cpu(mesh, source, {limits, sorted_index}, n_test);

    std::cout << "\n ptp_cpu_time= " << ptp_cpu_time << " (s)" << std::endl;

    double ptp_gpu_time =
        test_ptp_gpu(mesh, source, {limits, sorted_index}, n_test);

    std::cout << " ptp_gpu_time= " << ptp_gpu_time << " (s)" << std::endl;

    // FREE MEMORY
    delete mesh;
    delete[] toplesets;
    delete[] sorted_index;
}

double test_ptp_cpu(che*                   mesh,
                    const vector<index_t>& source,
                    const toplesets_t&     toplesets,
                    const int&             n_test)
{
    double t, seconds = INFINITY;

    distance_t* dist = new distance_t[mesh->n_vertices()];
    for (int i = 0; i < n_test; i++) {
        TIC(t)
        parallel_toplesets_propagation_cpu(dist, mesh, source, toplesets);
        TOC(t)
        seconds = min(seconds, t);
    }

    delete[] dist;

    return seconds;
}


double test_ptp_gpu(che*                   mesh,
                    const vector<index_t>& source,
                    const toplesets_t&     toplesets,
                    const int&             n_test)
{
    double t, seconds = INFINITY;

    distance_t* dist = new distance_t[mesh->n_vertices()];
    for (int i = 0; i < n_test; i++) {
        t = parallel_toplesets_propagation_coalescence_gpu(dist, mesh, source,
                                                           toplesets);
        seconds = min(seconds, t);
    }


    delete[] dist;

    return seconds;
}

}  // namespace gproshan
