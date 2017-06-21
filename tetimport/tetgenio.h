#pragma once
#define TINYOBJLOADER_IMPLEMENTATION

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <deque>
#include <random>
#include "Math.h"
#include"mesh_io.h"
#include "Intersections.h"
#include "tinyobj.h"

#define TETLIBRARY
#include "tetgen.h"

namespace XORShift { // XOR shift PRNG
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123; 
	inline float frand() { 
		unsigned int t;
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))) * (1.0f / 4294967295.0f); 
	}
}

std::random_device rd{};    
std::mt19937 engine{rd()};
std::uniform_real_distribution<double> distr{0.0, 1.0};



class tetrahedral_mesh
{
public:
	uint32_t oldnodenum, oldfacenum;
	std::deque<node>oldnodes;
	std::deque<face>oldfaces;

	uint32_t tetnum, nodenum, facenum;
	std::deque<tetrahedra>tetrahedras;
	std::deque<node>nodes;

	void loadobj(std::string filename);
};


void tetrahedral_mesh::loadobj(std::string filename)
{
	std::string mtldummy = "";
	tinyobj::attrib_t attrib;
std::vector<tinyobj::shape_t> shapes;
std::vector<tinyobj::material_t> materials;
  
std::string err;
bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(),mtldummy.c_str(), true);
// Loop over shapes
for (size_t s = 0; s < shapes.size(); s++) {
  // Loop over faces(polygon)
  size_t index_offset = 0;
  for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
    int fv = shapes[s].mesh.num_face_vertices[f];

    // Loop over vertices in the face.
    for (size_t v = 0; v < fv; v++) {
      // access to vertex
      tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
      tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
      tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
      tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
      tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
      tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
      tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];
      tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
      tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
    }
    index_offset += fv;

    // per-face material
    shapes[s].mesh.material_ids[f];
  }
}






	/*std::ifstream ifs(filename.c_str(), std::ifstream::in);
	if (!ifs.good())
	{
		std::cout << "Error loading obj:(" << filename << ") file not found!" << "\n";
		system("PAUSE");
		exit(0);
	}

	std::string line, key;
	int vertexid = 0, faceid = 0;
	std::cout << "Started loading obj file " << filename <<". \n";
	int linecounter = 0;
	while (!ifs.eof() && std::getline(ifs, line)) 
	{
	key = "";
	std::stringstream stringstream(line);
	stringstream >> key >> std::ws;

	if (key == "v") { // vertex	
		float x, y, z;
		while (!stringstream.eof())
		{
			stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
			oldnodes.push_back(node(vertexid, x, y, z));
			vertexid++;
		}
	}

	else if (key == "f") { // face
		int x,y,z;
		while (!stringstream.eof()) {
			stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
			oldfaces.push_back(face(faceid, x-1, y-1, z-1)); faceid++;
		}
	}
	linecounter++;
	if (linecounter == 100000) { std::cout << "Processed 100.000 lines\n"; linecounter = 0; }
	}*/

	float3 vertice;
	for (size_t v = 0; v < attrib.vertices.size() / 3; v++) 
	{
	  vertice.x = attrib.vertices[3 * v + 0];
	  vertice.y = attrib.vertices[3 * v + 1];
	  vertice.z = attrib.vertices[3 * v + 2];
	  oldnodes.push_back(node(v, vertice.x,vertice.y,vertice.z));
	}

	int32_t index_offset = 0;
	    // For each face
    for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++) {
      size_t fnum = shapes[0].mesh.num_face_vertices[f];

      // For each vertex in the face
      for (size_t v = 0; v < fnum; v++) {
        tinyobj::index_t idx = shapes[0].mesh.indices[index_offset + v];
      index_offset += fnum;
}



	tetgenio in, tmp, out;
	oldnodenum = attrib.vertices.size() / 3;
	oldfacenum = oldnodenum / 3;
	//oldnodenum = vertexid;
	//oldfacenum = oldfaces.size();

	/*in.pointlist = new REAL[in.numberofpoints * 3];
	for (int32_t i = 0; i < in.numberofpoints; i++)
	{
		in.pointlist[i * 3 + 0] = oldnodes.at(i).x;
		in.pointlist[i * 3 + 1] = oldnodes.at(i).y;
		in.pointlist[i * 3 + 2] = oldnodes.at(i).z;
	}*/

	// test 1 - get boundingbox and randomly generate points inside
	//=============================================================
	
	BBox mbox;
	mbox = init_BBox(&oldnodes);
	scale_BBox(mbox, 1.5); // scale boundingbox slightly bigger, so that wall triangles are well embedded into the tetrahedralization
	std::vector <float4> rndnodes;
	/*for (int i = 0; i < (int)(oldnodes.size()); i++)
	{
		float r1 = distr(engine);
		float r2 = distr(engine);
		float r3 = distr(engine);
		// NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
		// scale 0..1 values to BBox
		float n1 = ((r1 - 0) * (mbox.max.x - mbox.min.x) / (1 - 0)) + mbox.min.x;
		float n2 = ((r2 - 0) * (mbox.max.y - mbox.min.y) / (1 - 0)) + mbox.min.y;
		float n3 = ((r3 - 0) * (mbox.max.z - mbox.min.z) / (1 - 0)) + mbox.min.z;
		rndnodes.push_back(make_float4(n1, n2, n3, 0));
	} */
	rndnodes.push_back(make_float4(mbox.max.x, mbox.max.y, mbox.max.z, 0));
	rndnodes.push_back(make_float4(mbox.min.x, mbox.min.y, mbox.min.z, 0));
	rndnodes.push_back(make_float4(mbox.max.x, mbox.min.y, mbox.min.z, 0));
	rndnodes.push_back(make_float4(mbox.max.x, mbox.max.y, mbox.min.z, 0));
	rndnodes.push_back(make_float4(mbox.min.x, mbox.min.y, mbox.max.z, 0));
	rndnodes.push_back(make_float4(mbox.min.x, mbox.max.y, mbox.min.z, 0));
	rndnodes.push_back(make_float4(mbox.max.x, mbox.min.y, mbox.max.z, 0));
	rndnodes.push_back(make_float4(mbox.min.x, mbox.max.y, mbox.max.z, 0)); // 8 vertices of bounding box

	in.pointlist = new REAL[rndnodes.size() * 3];
	for (int32_t i = 0; i < rndnodes.size(); i++)
	{
		in.pointlist[i * 3 + 0] = rndnodes.at(i).x;
		in.pointlist[i * 3 + 1] = rndnodes.at(i).y;
		in.pointlist[i * 3 + 2] = rndnodes.at(i).z;
	}
	              
	in.numberofpoints = rndnodes.size();
	// create 6*2 triangle faces forming a bounding box cube
	// =====================================================
	in.numberoffacets = 12;
	in.facetlist = new tetgenio::facet[in.numberoffacets];
	tetgenio::facet *f;
	tetgenio::polygon *p;

	std::vector <int3> addfaces; // additional faces

	int32_t n1, n2, n3, n4, n5, n6, n7, n8;
	
	
	n1 = rndnodes.size() - 1;
	n2 = rndnodes.size() - 2;
	n3 = rndnodes.size() - 3;
	n4 = rndnodes.size() - 4;
	n5 = rndnodes.size() - 5;
	n6 = rndnodes.size() - 6;
	n7 = rndnodes.size() - 7;
	n8 = rndnodes.size() - 8;

	addfaces.push_back(make_int3(n1, n5, n6)); // 12 additional faces, each with three vertex ids
	addfaces.push_back(make_int3(n1, n2, n6));
	addfaces.push_back(make_int3(n3, n7, n8));
	addfaces.push_back(make_int3(n3, n4, n8));
	addfaces.push_back(make_int3(n5, n6, n7));
	addfaces.push_back(make_int3(n6, n7, n8));
	addfaces.push_back(make_int3(n1, n2, n3));
	addfaces.push_back(make_int3(n2, n3, n4));
	addfaces.push_back(make_int3(n1, n3, n5));
	addfaces.push_back(make_int3(n3, n5, n7));
	addfaces.push_back(make_int3(n2, n4, n6));
	addfaces.push_back(make_int3(n4, n6, n8));


	for (int i = 0; i < in.numberoffacets; i++)
	{

		f = &in.facetlist[i];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		p = &f->polygonlist[0];
		p->numberofvertices = 3;
		p->vertexlist = new int[p->numberofvertices]; // in vertexlist die nummer aus in.pointlist néhmen
		p->vertexlist[0] = addfaces.at(i).x;
		p->vertexlist[1] = addfaces.at(i).y;
		p->vertexlist[2] = addfaces.at(i).z;
	}

	fprintf(stderr, "Starting tetrahedralization..\n");
	tetrahedralize("nfznn", &in, &tmp); // 1st step - tetrahedralization of the vertices
	tmp.save_faces("tmp");
	tmp.save_elements("tmp");
	tmp.save_nodes("tmp");
	tmp.save_neighbors("tmp");
	tetrahedralize("rqnfnnzA", &tmp, &out); //2nd step - refinement of the mesh.
	out.save_faces("out");
	out.save_elements("out");
	out.save_nodes("out");
	out.save_neighbors("out");

	tetnum = out.numberoftetrahedra;
	facenum = out.numberoftrifaces;
	nodenum = out.numberofpoints;
	
	for (int i = 0; i < out.numberoftetrahedra; i++)
	{
		tetrahedra t;
		t.number = i;
		t.nindex1 = out.tetrahedronlist[i * 4 + 0];
		t.nindex2 = out.tetrahedronlist[i * 4 + 1];
		t.nindex3 = out.tetrahedronlist[i * 4 + 2];
		t.nindex4 = out.tetrahedronlist[i * 4 + 3];
		tetrahedras.push_back(t);
	}

	// assign neighbors to each tet
	for (int i = 0; i < out.numberoftetrahedra; i++)
	{
		tetrahedras.at(i).adjtet1 = out.neighborlist[i * 4 + 0];
		tetrahedras.at(i).adjtet2 = out.neighborlist[i * 4 + 1];
		tetrahedras.at(i).adjtet3 = out.neighborlist[i * 4 + 2];
		tetrahedras.at(i).adjtet4 = out.neighborlist[i * 4 + 3];
	}

	// assign faces to each tet
	for (int i = 0; i < out.numberoftetrahedra; i++)
	{
		tetrahedras.at(i).findex1 = out.tet2facelist[i * 4 + 0];
		tetrahedras.at(i).findex2 = out.tet2facelist[i * 4 + 1];
		tetrahedras.at(i).findex3 = out.tet2facelist[i * 4 + 2];
		tetrahedras.at(i).findex4 = out.tet2facelist[i * 4 + 3];
	}

	// assign tetrahedralization nodes to nodelist
	for (int i = 0; i < nodenum; i++)
	{
		float4 n = make_float4(out.pointlist[3 * i + 0], out.pointlist[3 * i + 1], out.pointlist[3 * i + 2], 0);
		nodes.push_back(node(i, n.x, n.y, n.z));
	}

	// set counter in each tetrahedron to zero
	for (int j = 0; j < out.numberoftetrahedra; j++)
	{
		tetrahedras.at(j).counter = 0;
	}
	for (int j = 0; j < oldfaces.size(); j++)
	{
		oldfaces.at(j).face_is_constrained = false;
	}

	// assign faces to tets
	fprintf_s(stderr, "Started assigning tets to faces...\n");
	for (int i = 0; i < oldfaces.size(); i++) // loop over all old faces
	{
		for (int j = 0; j < out.numberoftetrahedra; j++)  // for each face, loop over all tets
		{
			// faces
			int32_t n0 = oldfaces.at(i).node_a;
			int32_t n1 = oldfaces.at(i).node_b;
			int32_t n2 = oldfaces.at(i).node_c;
			float4 v0 = make_float4(oldnodes.at(n0).x, oldnodes.at(n0).y, oldnodes.at(n0).z, 0);
			float4 v1 = make_float4(oldnodes.at(n1).x, oldnodes.at(n1).y, oldnodes.at(n1).z, 0);
			float4 v2 = make_float4(oldnodes.at(n2).x, oldnodes.at(n2).y, oldnodes.at(n2).z, 0);

			// tets
			int32_t tn1 = tetrahedras.at(j).nindex1;
			int32_t tn2 = tetrahedras.at(j).nindex2;		
			int32_t tn3 = tetrahedras.at(j).nindex3;
			int32_t tn4 = tetrahedras.at(j).nindex4;

			float4 tv1 = make_float4(out.pointlist[3 * tn1 + 0], out.pointlist[3 * tn1 + 1], out.pointlist[3 * tn1 + 2], 0); 
			float4 tv2 = make_float4(out.pointlist[3 * tn2 + 0], out.pointlist[3 * tn2 + 1], out.pointlist[3 * tn2 + 2], 0);
			float4 tv3 = make_float4(out.pointlist[3 * tn3 + 0], out.pointlist[3 * tn3 + 1], out.pointlist[3 * tn3 + 2], 0);
			float4 tv4 = make_float4(out.pointlist[3 * tn4 + 0], out.pointlist[3 * tn4 + 1], out.pointlist[3 * tn4 + 2], 0);

			
			// tr_tri_intersect3D (double *C1, double *P1, double *P2, double *D1, double *Q1, double *Q2)

			int * coplanar = NULL; // dummy types
			float source[3];
			float target[3];


			float av0[3] = { v0.x, v0.y, v0.z };
			float ae1[3] = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
			float ae2[3] = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };
			float atv1[3] = { tv1.x, tv1.y, tv1.z };
			float atv2[3] = { tv2.x, tv2.y, tv2.z }; // 1,2,3		1,3,4		1,2,4		2,3,4

			float aev1[3] = { tv2.x - tv1.x, tv2.y - tv1.y, tv2.z - tv1.z };
			float aev2[3] = { tv3.x - tv1.x, tv3.y - tv1.y, tv3.z - tv1.z };

			float aev3[3] = { tv3.x - tv1.x, tv3.y - tv1.y, tv3.z - tv1.z };
			float aev4[3] = { tv4.x - tv1.x, tv4.y - tv1.y, tv4.z - tv1.z };

			float aev5[3] = { tv3.x - tv2.x, tv3.y - tv2.y, tv3.z - tv2.z };
			float aev6[3] = { tv4.x - tv2.x, tv4.y - tv2.y, tv4.z - tv2.z };

			if (tri_tri_intersection_test_3d(av0,ae1,ae2,atv1,aev1,aev2,coplanar,source,target) != 0 || tri_tri_intersection_test_3d(av0,ae1,ae2,atv1,aev3,aev4,coplanar,source,target) != 0 || tri_tri_intersection_test_3d(av0,ae1,ae2,atv1,aev1,aev4,coplanar,source,target) != 0 || tri_tri_intersection_test_3d(av0,ae1,ae2,atv2,aev5,aev6,coplanar,source,target) != 0)
			{
				oldfaces.at(i).face_is_constrained = true;
				tetrahedras.at(j).hasfaces = true;
				tetrahedras.at(j).faces[tetrahedras.at(j).counter] = i; // tetrahedron at position 'j' gets face at 'i' assigned 
				tetrahedras.at(j).counter = tetrahedras.at(j).counter + 1; // increase counter 
			}

		}
		if (i == (int)oldfaces.size()/4) fprintf_s(stderr, "25%% done\n");
		if (i == (int)oldfaces.size()/2) fprintf_s(stderr, "50%% done\n");
		if (i == (int)oldfaces.size()*3/4) fprintf_s(stderr, "75%% done\n");
	}


	for (auto f : oldfaces)
	{
		if (!f.face_is_constrained)
		{
			fprintf_s(stderr, "Error while assigning faces!\n");
			system("PAUSE");
			exit(0);
		}

	}
	fprintf_s(stderr, "Finished assigning faces to tets!\n");

}


