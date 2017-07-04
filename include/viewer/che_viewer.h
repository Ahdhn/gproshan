#ifndef CHE_VIEWER_H
#define CHE_VIEWER_H

#include "che.h"

#include <GL/freeglut.h>

#define COLOR 0.5

#ifdef SINGLE_P
#define glVertex3v(x) glVertex3fv(x)
#define GL_VERTEX_T GL_FLOAT
#else
#define glVertex3v(x) glVertex3dv(x)
#define GL_VERTEX_T GL_DOUBLE
#endif

typedef vertex_t color_t; 

class che_viewer
{
	protected:
		che * mesh;
		size_t _n_vertices; // current number of vertices
		bool invert_orientation;
		vertex * normals;
		color_t * colors;
		GLuint vao;
		GLuint vbo[4];

	public:
		che_viewer();
		virtual ~che_viewer();
		che *& operator -> ();
		operator che *& ();
		void init(che * _mesh);
		void update();
		void update_vbo();
		void update_normals();
		void update_colors(const color_t *const c = NULL);
		void draw();
		
		const size_t & n_vertices() const;
		color_t & color(const index_t & v);
		vertex & normal(const index_t & v);
};

#endif // CHE_VIEWER_H
