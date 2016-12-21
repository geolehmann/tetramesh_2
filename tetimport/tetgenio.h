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
	uint32_t oldnodenum, oldfacenum;
	std::deque<node>oldnodes;
	std::deque<face>oldfaces;

	uint32_t tetnum, nodenum, facenum;
	std::deque<tetrahedra>tetrahedras;
	std::deque<node>nodes;
	std::deque<face>faces;

	std::deque<uint32_t> adjfaces_num;
	std::deque<uint32_t> adjfaces_numlist;

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
	}

	tetgenio in, tmp, out;
	in.numberofpoints = vertexid;

	oldnodenum = in.numberofpoints;
	oldfacenum = oldfaces.size();

	in.pointlist = new REAL[in.numberofpoints * 3];
	for (int32_t i = 0; i < in.numberofpoints; i++)
	{
		in.pointlist[i * 3 + 0] = oldnodes.at(i).x;
		in.pointlist[i * 3 + 1] = oldnodes.at(i).y;
		in.pointlist[i * 3 + 2] = oldnodes.at(i).z;
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

	// set in each tetrahedron the counter to zero
	for (int j = 0; j < out.numberoftetrahedra; j++)
	{
		tetrahedras.at(j).counter = 0;
	}

	// assign faces to tets
	fprintf_s(stderr, "Started assigning tets to faces...\n");
	for (int i = 0; i < oldfaces.size(); i++) // loop over all faces
	{
		for (int j = 0; j < out.numberoftetrahedra; j++)  // for each face, loop over all tets
		{
			int32_t n0 = oldfaces.at(i).node_a;
			int32_t n1 = oldfaces.at(i).node_b;
			int32_t n2 = oldfaces.at(i).node_c;
			float4 v0 = make_float4(oldnodes.at(n0).x, oldnodes.at(n0).y, oldnodes.at(n0).z, 0);
			float4 v1 = make_float4(oldnodes.at(n1).x, oldnodes.at(n1).y, oldnodes.at(n1).z, 0);
			float4 v2 = make_float4(oldnodes.at(n2).x, oldnodes.at(n2).y, oldnodes.at(n2).z, 0);
			int32_t tn1 = tetrahedras.at(j).nindex1;
			int32_t tn2 = tetrahedras.at(j).nindex2;
			int32_t tn3 = tetrahedras.at(j).nindex3;
			int32_t tn4 = tetrahedras.at(j).nindex4;
			float4 tv1 = make_float4(out.pointlist[3 * tn1 + 0], out.pointlist[3 * tn1 + 1], out.pointlist[3 * tn1 + 2], 0); // prüfen ob mit nodes. ersetzen
			float4 tv2 = make_float4(out.pointlist[3 * tn2 + 0], out.pointlist[3 * tn2 + 1], out.pointlist[3 * tn2 + 2], 0);
			float4 tv3 = make_float4(out.pointlist[3 * tn3 + 0], out.pointlist[3 * tn3 + 1], out.pointlist[3 * tn3 + 2], 0);
			float4 tv4 = make_float4(out.pointlist[3 * tn4 + 0], out.pointlist[3 * tn4 + 1], out.pointlist[3 * tn4 + 2], 0);// now we have the four vertices of the tetrahedron

			if (RayTetIntersectionCPU(v0, v1, tv1, tv2, tv3, tv4) || RayTetIntersectionCPU(v0, v2, tv1, tv2, tv3, tv4) || RayTetIntersectionCPU(v1, v2, tv1, tv2, tv3, tv4))
			{
				tetrahedras.at(j).hasfaces = true;
				// check if face is already in array
				bool alreadythere = false;
				for (int k = 0; k < 99; k++)
				{
					if (tetrahedras.at(j).faces[k] == i) alreadythere = true;
				}
				if (!alreadythere) 
				{
					tetrahedras.at(j).faces[tetrahedras.at(j).counter] = i; // tetrahedron at position 'j' gets face at 'i' assigned 
					tetrahedras.at(j).counter = tetrahedras.at(j).counter + 1; // increase counter 
				}
			}

		}
		if (i == (int)oldfaces.size()/4) fprintf_s(stderr, "25%% done\n");
		if (i == (int)oldfaces.size()/2) fprintf_s(stderr, "50%% done\n");
		if (i == (int)oldfaces.size()*4/3) fprintf_s(stderr, "75%% done\n");
	}
	fprintf_s(stderr, "Finished assigning faces to tets!\n");

	//=====================================================================================================================

	uint32_t currentindex = 0;
	for (auto ctet : tetrahedras) // loop over all tetrahedra
	{
		for (int i = 0; i < ctet.counter;i++)
		{
			if (ctet.faces[i] !=0) adjfaces_numlist.push_back(ctet.faces[i]);
		}
		adjfaces_num.push_back(currentindex + ctet.counter); // in adjfaces_num sind pro tet die anzahl der faces
		currentindex += ctet.counter;
	}
	fprintf_s(stderr, "Finished mesh preparation! \n");
}


