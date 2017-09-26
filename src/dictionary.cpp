#include "dictionary.h"

#include "sampling.h"
#include "d_dict_learning.h"

#include <cassert>

const size_t dictionary::min_nvp = 36;
const size_t dictionary::L = 10;

dictionary::dictionary(che *const & _mesh, basis *const & _phi_basis, const size_t & _m, const size_t & _M, const distance_t & f, const bool & _d_plot):
					mesh(_mesh), phi_basis(_phi_basis), m(_m), M(_M), d_plot(_d_plot)
{
	n_vertices = mesh->n_vertices();

	// load sampling
	if(M == 0)
	{
		M = mesh->n_vertices();
		phi_basis->radio = 3 * mesh->mean_edge();
	}
	else
	{
		sampling.reserve(M);
		assert(load_sampling(sampling, phi_basis->radio, mesh, M));
	}
	
	s_radio = phi_basis->radio;
	phi_basis->radio *= f;

	patches.resize(M);
	patches_map.resize(n_vertices);
	
	patch_t::del_index = false;
	init_patches();

	A.eye(phi_basis->dim, m);
	alpha.zeros(m, M);

	if(d_plot) phi_basis->plot_basis();
}

dictionary::~dictionary()
{
	patch_t::del_index = true;
}

void dictionary::learning()
{
	string f_dict = "tmp/" + mesh->name() + '_' + to_string(phi_basis->dim) + '_' + to_string(m) + ".dict";
	debug(f_dict)

	if(!A.load(f_dict))
	{
		A.eye(phi_basis->dim, m);
		// A.random(phi_basis->dim, m);

		d_message(dictionary learning...)
		TIC(d_time) KSVDT(A, patches, M, L); TOC(d_time)
		debug(d_time)

		A.save(f_dict);
	}

	assert(A.n_rows == phi_basis->dim);
	assert(A.n_cols == m);

	if(d_plot) phi_basis->plot_atoms(A);
}

void dictionary::denoising()
{
	d_message(sparse coding...)
	TIC(d_time)
	OMP_all_patches_ksvt(alpha, A, patches, M, L);
	TOC(d_time)

	d_message(mesh reconstruction...)
	assert(n_vertices == mesh->n_vertices());

	TIC(d_time)
	mesh_reconstruction(mesh, M, patches, patches_map, A, alpha);
	TOC(d_time)
	debug(d_time)
}

void dictionary::init_patches()
{
	#pragma omp parallel for
	for(index_t s = 0; s < M; s++)
	{
		index_t v = sample(s);
		patch_t & p = patches[s];

		geodesics fm(mesh, {v}, NIL, phi_basis->radio);

		p.n = fm.get_n_radio();
		p.indexes = new index_t[p.n];
		fm.get_sort_indexes(p.indexes, p.n);
	}
	
	for(index_t s = 0; s < M; s++)
	{
		patch_t & p = patches[s];
		p.reset_xyz(mesh, patches_map, s);
	}

	#pragma omp parallel for
	for(index_t s = 0; s < M; s++)
	{
		patch_t & p = patches[s];
		
		assert(p.n > min_nvp); // old code change to principal_curvatures
		jet_fit_directions(p);

		p.transform();
		p.phi.set_size(p.n, phi_basis->dim);
		phi_basis->discrete(p.phi, p.xyz);
	}

	#ifndef NDEBUG
		size_t patch_mean_size = 0;
	
		#pragma omp parallel for reduction(+: patch_mean_size)
		for(index_t s = 0; s < M; s++)
			patch_mean_size += patches[s].n;
		
		patch_mean_size /= M;
		debug(patch_mean_size)
	#endif
}

index_t dictionary::sample(const index_t & s)
{
	assert(s < M);
	if(sampling.size()) return sampling[s];
	return s;
}
