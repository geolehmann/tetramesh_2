#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <deque>
#include "Math.h"
#include"mesh_io.h"

#define TETLIBRARY
#include "tetgen.h"

class tetrahedral_mesh
{
public:
	uint32_t tetnum, nodenum, facenum, edgenum;
	std::deque<tetrahedra>tetrahedras;
	std::deque<node>nodes;
	std::deque<face>oldfaces;

	std::deque<uint32_t> adjfaces_num;
	std::deque<uint32_t> adjfaces_numlist;

	std::deque<uint32_t> adjfaces_list; // temporary list

	std::deque<face>faces;
	std::deque<edge>edges;
	uint32_t max = 1000000000;

	void loadobj(std::string filename);
};

void tetrahedral_mesh::loadobj(std::string filename)
{
	std::ifstream ifs(filename.c_str(), std::ifstream::in);
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
			nodes.push_back(node(vertexid, x, y, z)); vertexid++;
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
	}

	tetgenio in, tmp, out;
	in.numberofpoints = vertexid - 1;

	nodenum = in.numberofpoints + 1;
	facenum = nodenum / 3;

	in.pointlist = new REAL[in.numberofpoints * 3];
	for (int32_t i = 0; i < in.numberofpoints; i++)
	{
		in.pointlist[i * 3 + 0] = nodes.at(i).x;
		in.pointlist[i * 3 + 1] = nodes.at(i).y;
		in.pointlist[i * 3 + 2] = nodes.at(i).z;
	}
	fprintf(stderr, "Starting tetrahedralization..\n");
	tetrahedralize("nfznn", &in, &tmp); // first tetrahedralization of the vertices
	tmp.save_faces("tmp");
	tmp.save_elements("tmp");
	tmp.save_nodes("tmp");
	tmp.save_neighbors("tmp");
	tetrahedralize("rqnfnnzA", &tmp, &out); //2nd step - refinement of the mesh.
	out.save_faces("out");
	out.save_elements("out");
	out.save_nodes("out");
	out.save_neighbors("out");

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
	tetnum = out.numberoftetrahedra;
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

	// assign nodes to faces
	for (int i = 0; i < oldfaces.size(); i++)
	{
		for (int j = 0; j < out.numberoftetrahedra; j++)
		{
			int32_t n0 = oldfaces.at(i).node_a;
			int32_t n1 = oldfaces.at(i).node_b;
			int32_t n2 = oldfaces.at(i).node_c;
			float4 v0 = make_float4(nodes.at(n0).x, nodes.at(n0).y, nodes.at(n0).z, 0);
			float4 v1 = make_float4(nodes.at(n1).x, nodes.at(n1).y, nodes.at(n1).z, 0);
			float4 v2 = make_float4(nodes.at(n2).x, nodes.at(n2).y, nodes.at(n2).z, 0);
			Ray r1 = Ray(v0, v1 - v0);
			Ray r2 = Ray(v0, v2 - v0);
			Ray r3 = Ray(v1, v2 - v1); // now we have the three edges of the current face as rays




		}
	}
	
}


