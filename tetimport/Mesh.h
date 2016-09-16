#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include "Math.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define TETLIBRARY
#include "tetgen.h"

//----------------------- tetgen interop -----------------------------------------------------------------

void tetrahedralize_nodes(std::string inputfile, mesh3* mesh)
{
	std::vector<float3> nodes;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inputfile.c_str());
	if (!err.empty()) { std::cerr << err << std::endl; }
	if (!ret) { exit(1); }

	std::cout << "# of shapes    : " << shapes.size() << std::endl;
	std::cout << "# of materials : " << materials.size() << std::endl;


	for (int32_t i = 0; i < attrib.vertices.size() / 3; i++) // loop over all vertices, indepedently from number of shapes
	{
		float vx = attrib.vertices[3 * i + 0];
		float vy = attrib.vertices[3 * i + 1];
		float vz = attrib.vertices[3 * i + 2];
		nodes.push_back(make_float3(vx, vy, vz));
	}


	// 2. put nodes into tetgenio->in and tetrahedralize
	tetgenio in, out;
	in.firstnumber = 1;  // All indices start from 1.
	in.numberofpoints = (int)(attrib.vertices.size() / 3); // number of nodes
	in.pointlist = new REAL[attrib.vertices.size()];
	for (int i = 0; i < in.numberofpoints; i++)
	{
		// get nodes into pointlist
		in.pointlist[i * 3 + 0] = nodes.at(i).x;
		in.pointlist[i * 3 + 1] = nodes.at(i).y;
		in.pointlist[i * 3 + 2] = nodes.at(i).z;
	}
	tetrahedralize("fn-nn", &in, &out); // parameter Q for quiet mode

//----------------some debug stuff:--------------------------------------------------->
	std::ofstream f1;
	f1.open("point2tetlist.txt", std::ios_base::app);

	for (int32_t i = 0; i < out.numberoftetrahedra; i++)
	{
		int t = out.neighborlist[i];
		f1 << t << std::endl;
	}

//<----------------some debug stuff:---------------------------------------------------

	//out - pointlist, tetrahedronlist, trifacelist, neighborlist, tet2facelist, face2tetlist
	  // 'neighborlist':  An array of tetrahedron neighbors; 4 ints per element. 
  // 'tet2facelist':  An array of tetrahedron face indices; 4 ints per element.
  // 'tet2edgelist':  An array of tetrahedron edge indices; 6 ints per element.


	// 3. put tetmesh into 'mesh' structure, connect faces from tinyobj to mesh3
	mesh->facenum = out.numberoftrifaces;
	mesh->nodenum = out.numberofpoints;
	mesh->tetnum = out.numberoftetrahedra;

	for (int i = 0; i < out.numberofpoints; i++)
	{
		// get nodes from pointlist to mesh3
		float a = out.pointlist[i * 3 + 0];
		float b = out.pointlist[i * 3 + 1];
		float c = out.pointlist[i * 3 + 2];
		mesh->n_x.push_back(a);
		mesh->n_y.push_back(b);
		mesh->n_z.push_back(c);
		mesh->n_index.push_back(i);
	}

	for (int i = 0; i < out.numberoftetrahedra / 4; i++)
	{
		mesh->t_nindex1.push_back(out.tetrahedronlist[i * 4 + 0]);
		mesh->t_nindex2.push_back(out.tetrahedronlist[i * 4 + 1]);
		mesh->t_nindex3.push_back(out.tetrahedronlist[i * 4 + 2]);
		mesh->t_nindex4.push_back(out.tetrahedronlist[i * 4 + 3]);
	}


	// this part for association of faces to tetnodes
	std::vector<std::pair <int32_t, int32_t>> matches;
	int counter = 0;
	for (int32_t i = 0; i < attrib.vertices.size() / 3; i++)
	{
		float vx = attrib.vertices[3 * i + 0]; // these are the vertices of the original mesh
		float vy = attrib.vertices[3 * i + 1];
		float vz = attrib.vertices[3 * i + 2];
		float a, b, c;
		for (int32_t j = 0; j < out.numberofpoints / 3; j++)
		{
			// these are the vertices of the generated tetmesh
			a = out.pointlist[j * 3 + 0];
			b = out.pointlist[j * 3 + 1];
			c = out.pointlist[j * 3 + 2];


			if (nearlysame(vx, a) && nearlysame(vy, b) && nearlysame(vz, c))
			{
				//its a MATCH!!!
				matches.push_back(std::make_pair(i, j));
				counter++;
			}

		}
	}
	fprintf(stderr, "Done matching faces\n\n");

}
