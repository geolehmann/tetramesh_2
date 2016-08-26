#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "Math.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define TETLIBRARY
#include "tetgen.h"

//----------------------- tetgen interop -----------------------------------------------------------------

void tetrahedralize_nodes(std::string inputfile, tetgenio &out)
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

	for (int32_t i = 0; i < attrib.vertices.size() / 3; i++)
	{
		float vx = attrib.vertices[3 * i + 0];
		float vy = attrib.vertices[3 * i + 1];
		float vz = attrib.vertices[3 * i + 2];
		nodes.push_back(make_float3(vx, vy, vz));
	}

	// 2. put nodes into tetgenio->in and tetrahedralize
	tetgenio in, tmp;
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
	tetrahedralize("Q", &in, &out); // parameter Q for quiet mode
}
