#ifndef TEST_GEODESICS_PTP_H
#define TEST_GEODESICS_PTP_H

#include "geodesics_ptp.h"


// geometry processing and shape analysis framework
namespace gproshan {


/// Execute performance and accuracy test for ptp algorithm on cpu and gpu.
void main_test_geodesics_ptp(const int& nargs, const char** args);

double test_ptp_cpu(che*                        mesh,
                    const std::vector<index_t>& source,
                    const toplesets_t&          toplesets,
                    const int&                  n_test,
                    distance_t*                 dist);

double test_ptp_gpu(che*                        mesh,
                    const std::vector<index_t>& source,
                    const toplesets_t&          toplesets,
                    const int&                  n_test,
                    distance_t*                 dist);


/// Exact geodesics computed using MeshLP
/// https://github.com/areslp/matlab/tree/master/MeshLP/MeshLP, Geodesics code:
/// http://code.google.com/p/geodesic/


}  // namespace gproshan

#endif  // TEST_GEODESICS_PTP_H
