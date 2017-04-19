#include "laplacian.h"

void laplacian(che * mesh, sp_mat & L, sp_mat & A)
{
	size_t n_edges = mesh->n_edges();
	size_t n_vertices = mesh->n_vertices();

	umat DI(2, 2 * n_edges);
	vec DV(2 * n_edges);

	umat SI(2, n_edges);
	vec SV(n_edges);

	#pragma omp parallel for
	for(index_t e = 0; e < n_edges; e++)
	{
		index_t i = e << 1;

		DI(0, i) = e;
		DI(1, i) = mesh->vt(mesh->et(e));
		DV(i) = 1;

		i++;

		DI(0, i) = e;
		DI(1, i) = mesh->vt(next(mesh->et(e)));
		DV(i) = -1;

		SI(0, e) = SI(1, e) = e;
		SV(e) = (mesh->cotan(mesh->et(e)) + 
					mesh->cotan(mesh->ot_et(e))) / 2;
	}

	sp_mat D(DI, DV, n_edges, n_vertices);
	sp_mat S(SI, SV, n_edges, n_edges);
	
	L = D.t() * S * D;

	A.resize(n_vertices, n_vertices);
	for(index_t v = 0; v < n_vertices; v++)
		A(v, v) = mesh->area_vertex(v);
}

void laplacian(che * mesh, sp_mat_e & L, sp_mat_e & A)
{
	size_t n_edges = mesh->n_edges();
	size_t n_vertices = mesh->n_vertices();

	sp_mat_e D(n_edges, n_vertices);
	sp_mat_e S(n_edges, n_edges);

	D.reserve(VectorXi::Constant(n_edges,2));
	S.reserve(VectorXi::Constant(n_edges,1));

	for(index_t e = 0; e < n_edges; e++)
	{
		D.insert(e, mesh->vt(mesh->et(e))) = 1;
		D.insert(e, mesh->vt(next(mesh->et(e)))) = -1;

		S.insert(e, e) = (mesh->cotan(mesh->et(e)) + 
					mesh->cotan(mesh->ot_et(e))) / 2;
	}

	L = D.transpose() * S * D;

	A.resize(n_vertices, n_vertices);
	for(index_t v = 0; v < n_vertices; v++)
		A.insert(v, v) = mesh->area_vertex(v);
	
}

void eigs_laplacian(vec & eigval, mat & eigvec, che * mesh, const sp_mat & L, const size_t & K)
{
	string feigval = "tmp/" + mesh->name() + '_' + to_string(K) + ".a_eigval";
	string feigvec = "tmp/" + mesh->name() + '_' + to_string(K) + ".a_eigvec";
	
	debug(feigval)
	debug(feigvec)
	
	if(!eigval.load(feigval) || !eigvec.load(feigvec))
	{
		eigs_sym(eigval, eigvec, L, K, "sm");
		eigval.save(feigval);
		eigvec.save(feigvec);
	}
}

