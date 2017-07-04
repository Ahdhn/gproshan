#include "che_viewer.h"

#include <GLES3/gl3.h>

che_viewer::che_viewer(che * _mesh)
{
	n_vertices = 0;
	mesh = _mesh;
	invert_orientation = false;
	
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(4, vbo);

	update();
}

che_viewer::~che_viewer()
{
	glDeleteBuffers(4, vbo);  
	glDeleteVertexArrays(1, &vao);
	
	delete [] normals;
	delete [] colors;
}

che_viewer::operator che *const ()
{
	return mesh;
}

void che_viewer::update()
{
	if(n_vertices != mesh->n_vertices())
	{
		delete [] normals;
		delete [] colors;

		n_vertices = mesh->n_vertices();
		normals = new vertex[n_vertices];
		colors = new color_t[n_vertices];

		update_normals();
		update_colors();
	}

	update_vbo();
}

void che_viewer::update_vbo()
{
	// 0 VERTEX
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, mesh->n_vertices() * sizeof(vertex), &mesh->gt(0), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_VERTEX_T, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 1 NORMAL
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, mesh->n_vertices() * sizeof(vertex), normals, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_VERTEX_T, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	// 2 COLOR
	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, mesh->n_vertices() * sizeof(vertex_t), colors, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_VERTEX_T, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// INDEXES
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[3]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->n_half_edges() * sizeof(index_t), &mesh->vt(0), GL_STATIC_DRAW);
//	glEnableVertexAttribArray(3);
//	glVertexAttribPointer(3, 3, GL_UNSIGNED_INT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void che_viewer::update_normals()
{
	#pragma omp parallel for
	for(index_t v = 0; v < n_vertices; v++)
	{
		normals[v] = mesh->normal(v);
		if(invert_orientation) normals[v] = -normals[v];
	}
}

void che_viewer::update_colors(const color_t *const c)
{
	#pragma omp parallel for
	for(index_t v = 0; v < n_vertices; v++)
		colors[v] = c ? c[v] : COLOR;
}

void che_viewer::draw()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[3]);
	glDrawElements(GL_TRIANGLES, mesh->n_half_edges(), GL_UNSIGNED_INT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

