/*
*  tetrahedra-based raytracer
*  Copyright (C) 2015-2016  Christian Lehmann
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*/

#define GLEW_STATIC
#include "Util.h"
#include "tetgenio.h"
#include "Camera.h"
#include "device_launch_parameters.h"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Sphere.h"
#include "OpenFileDialog.h"

#define spp 1
#define gamma 2.2f


float3* finalimage;
float3* accumulatebuffer;
uint32_t frameNumber = 0;
bool bufferReset = false;
float deltaTime, lastFrame;
BBox box;
GLuint vbo;
mesh2 *mesh;
__managed__ bool edgeVisualization = false;
__managed__ bool distVisualization = false;
__managed__ int MAX_DEPTH = 3;
__managed__ int width = 1920;
__managed__ int height = 1080;

// Camera
InteractiveCamera* interactiveCamera = NULL;
Camera* hostRendercam = NULL;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
int lastX;
int lastY;
bool buttonActive = false, enableMouseMovement = true, cursorFree = false;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = 0.0;
int theButtonState = 0;
int theModifierState = 0;
float scalefactor = 1.2f;

union Color  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// CUDA error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("PAUSE");
		if (abort) exit(code);
	}
}

unsigned int WangHash(unsigned int a) {
	// richiesams.blogspot.co.nz/2015/03/creating-randomness-and-acummulating.html
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

static void error_callback(int error, const char* description)
{
	// GLFW error callback
	fputs(description, stderr);
}

void updateCamPos()
{
	float4 pos = hostRendercam->position;
	// check if current pos is still inside tetrahedralization
	ClampToBBox(&box, hostRendercam->position);
	int32_t adjtets[4] = { mesh->t_adjtet1[_start_tet], mesh->t_adjtet2[_start_tet], mesh->t_adjtet3[_start_tet], mesh->t_adjtet4[_start_tet] };
	if (!IsPointInThisTetCPU(mesh, pos, _start_tet))
	{
	//fprintf(stderr, "Alert - Outside \n");
	//fprintf(stderr, "Adjacent tets: %ld %ld %ld %ld  \n", adjtets[0], adjtets[1], adjtets[2], adjtets[3]);
	if (IsPointInThisTetCPU(mesh, pos, adjtets[0])) _start_tet = adjtets[0];
	else if (IsPointInThisTetCPU(mesh, pos, adjtets[1])) _start_tet = adjtets[1];
	else if (IsPointInThisTetCPU(mesh, pos, adjtets[2])) _start_tet = adjtets[2];
	else if (IsPointInThisTetCPU(mesh, pos, adjtets[3])) _start_tet = adjtets[3];
	else
	{
		fprintf(stderr, "Fallback to CUDA search for starting tet\n");
		uint32_t _dim = 2 + pow(mesh->tetnum, 0.25);
		dim3 Block(_dim, _dim, 1);
		dim3 Grid(_dim, _dim, 1);
		GetTetrahedraFromPoint << <Grid, Block >> >(mesh, pos);
		gpuErrchk(cudaDeviceSynchronize());
	}
	//fprintf(stderr, "New starting tet: %ld \n", _start_tet);
	}
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	float dist = 0.3; // skipping tetras if set too high...

	if (action == GLFW_PRESS) buttonActive = true;
	if (action == GLFW_RELEASE) buttonActive = false;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_A && buttonActive)
	{
		interactiveCamera->strafe(-dist); updateCamPos();
	}
	if (key == GLFW_KEY_D && buttonActive)
	{
		interactiveCamera->strafe(dist); updateCamPos();
	}
	if (key == GLFW_KEY_W && buttonActive)
	{
		interactiveCamera->goForward(dist); updateCamPos();
	}
	if (key == GLFW_KEY_S && buttonActive)
	{
		interactiveCamera->goForward(-dist); updateCamPos();
	}

	if (key == GLFW_KEY_R && buttonActive)
	{
		interactiveCamera->changeAltitude(dist); updateCamPos();
	}
	if (key == GLFW_KEY_F && buttonActive)
	{
		interactiveCamera->changeAltitude(-dist); updateCamPos();
	}
	if (key == GLFW_KEY_G && buttonActive)
	{
		interactiveCamera->changeApertureDiameter(0.1);
	}
	if (key == GLFW_KEY_H && buttonActive)
	{
		interactiveCamera->changeApertureDiameter(-0.1);
	}
	if (key == GLFW_KEY_T && buttonActive)
	{
		interactiveCamera->changeFocalDistance(0.1);
	}
	if (key == GLFW_KEY_Z && buttonActive)
	{
		interactiveCamera->changeFocalDistance(-0.1);
	}

	if (key == GLFW_KEY_UP && buttonActive)
	{
		interactiveCamera->changePitch(0.02f);
	}
	if (key == GLFW_KEY_DOWN && buttonActive)
	{
		interactiveCamera->changePitch(-0.02f);
	}
	if (key == GLFW_KEY_LEFT && buttonActive)
	{
		interactiveCamera->changeYaw(0.02f);
	}
	if (key == GLFW_KEY_RIGHT && buttonActive)
	{
		interactiveCamera->changeYaw(-0.02f);
	}
	if (key == GLFW_KEY_B && buttonActive)
	{
		// debug stuff
		updateCamPos();
	}
	if (key == GLFW_KEY_M && action == GLFW_PRESS)
	{
		// debug stuff
		if (!edgeVisualization) edgeVisualization = true; else edgeVisualization = false;
	}
	if (key == GLFW_KEY_V && action == GLFW_PRESS)
	{
		// debug stuff
		if (!distVisualization) distVisualization = true; else distVisualization = false;
	}
	if (key == GLFW_KEY_C && action == GLFW_PRESS)
	{
		if (cursorFree == false) { glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); cursorFree = true; enableMouseMovement = false; }
		else { glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); cursorFree = false; enableMouseMovement = true; }
	}

	bufferReset = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) theButtonState = 0;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) theButtonState = 1;
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) theButtonState = 2;
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	int deltaX = lastX - xpos;
	int deltaY = lastY - ypos;

	if (enableMouseMovement) if (deltaX != 0 || deltaY != 0) {

		if (theButtonState == 0)  // Rotate
		{
			interactiveCamera->changeYaw(deltaX * 0.01);
			interactiveCamera->changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == 1) // Zoom
		{
			interactiveCamera->changeAltitude(-deltaY * 0.01);
			updateCamPos();
		}

		if (theButtonState == 2) // camera move
		{
			interactiveCamera->changeRadius(-deltaY * 0.01);
			updateCamPos();
		}

		lastX = xpos;
		lastY = ypos;
		bufferReset = true;
	}
}

__device__ RGB radiance(mesh2 *mesh, int32_t start, float4 &rayo, float4 &rayd, float4 oldpos, curandState* randState, float &distance)
{
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 1.0f);	// colour mask (accumulated reflectance)
	float4 accucolor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	// accumulated colour
	float4 originInWorldSpace = rayo;
	float4 rayInWorldSpace = rayd;
	int32_t newstart = start;

	for (int bounces = 0; bounces < MAX_DEPTH; bounces++)
	{
		float4 f = make_float4(0, 0, 0, 0);  // primitive colour
		float4 emit = make_float4(0, 0, 0, 0); // primitive emission colour
		float4 x; // intersection point
		float4 n; // normal
		float4 nl; // oriented normal
		float4 dw; // ray direction of next path segment
		float4 pointHitInWorldSpace;
		float3 rayorig = make_float3(originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z);
		float3 raydir = make_float3(rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z);
		bool isEdge = false;
		double dist=DBL_MAX;
		rayhit firsthit;
		Geometry geom;


		// ------------------------------ TRIANGLE intersection --------------------------------------------
		
		
		// test - loop over all triangles, test for intersection - klappt!!!
		/*for (int i = 0; i < mesh->oldfacenum; i++)
		{ // fg_node und ng_x stimmen!!!!!!!!!!!!!!!!
					int32_t na = mesh->fg_node_a[i];
					int32_t nb = mesh->fg_node_b[i];
					int32_t nc = mesh->fg_node_c[i];
					float4 v1 = make_float4(mesh->ng_x[na], mesh->ng_y[na], mesh->ng_z[na], 0);
					float4 v2 = make_float4(mesh->ng_x[nb], mesh->ng_y[nb], mesh->ng_z[nb], 0);
					float4 v3 = make_float4(mesh->ng_x[nc], mesh->ng_y[nc], mesh->ng_z[nc], 0);
					float d_new =  RayTriangleIntersection(Ray(originInWorldSpace, rayInWorldSpace), v1, v2, v3);
					if (d_new < dist && d_new > 0.0001) 
					{ 
						dist = d_new; firsthit.constrained = true; 		
						firsthit.face = i;
						float4 e1 = v2 - v1;
						float4 e2 = v3 - v1;
						float4 s = originInWorldSpace - v1;
						n = Cross(e1, e2);
					}
		}*/

		//dist = 9999; firsthit.face = -9999;*/
		traverse_ray(mesh, originInWorldSpace, rayInWorldSpace, newstart, firsthit, dist, edgeVisualization, isEdge, n);

		distance = dist;

		pointHitInWorldSpace = originInWorldSpace + rayInWorldSpace * dist;

		// ------------------------------ SPHERE intersection --------------------------------------------
		float4 spherePos = make_float4(10,10,10,0);
		float sphereRad = 10, sphereDist = 0;
		bool spheresEnabled = false;	
		if (spheresEnabled) { sphereDist = sphIntersect(originInWorldSpace, rayInWorldSpace, spherePos, sphereRad); }
		if (sphereDist > 0.0) {	geom = SPHERE; traverse_until_point(mesh, originInWorldSpace, rayInWorldSpace, newstart, originInWorldSpace + rayInWorldSpace * sphereDist, firsthit); }
		else { geom = TRIANGLE; }

		if (geom == SPHERE)
		{
		emit = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		f = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
		firsthit.refl_t = REFR;
		x = originInWorldSpace + rayInWorldSpace * sphereDist;
		float4 ds = x - spherePos;
		n= normalize(ds);
		nl = Dot(n, rayInWorldSpace) < 0 ? n : n * -1;
		}

		if (geom == TRIANGLE)
		{
			x = pointHitInWorldSpace;
			n = normalize(n);
			nl = Dot(n, rayInWorldSpace) < 0 ? n : n * -1;

			/*if (firsthit.constrained == true) { emit = make_float4(1.0f, 0.0f, 0.3f, 0.0f); f = make_float4(0.75f, 0.75f, 0.75f, 0.0f); } // blue is constrained

			if (firsthit.wall == true) 
			{ 
				emit = make_float4(10.0f, 10.0f, 0.4f, 0.0f); // wall wird erkannt
				f = make_float4(0.3f, 0.1f, 0.4f, 0.0f); 
				/*float4 color1 = make_float4(0, 0, 0, 0);
				float4 color2 = make_float4(0.0f, 1.0f, 1.0f, 0);
				float percent = (((rayInWorldSpace.y + 1) * (1 - 0)) / (1 + 1)) + 0;
				float red = color1.x + percent * (color2.x - color1.x);
				float green = color1.y + percent * (color2.y - color1.y);
				float blue = color1.z + percent * (color2.z - color1.z);
				f = make_float4(red, green, blue, 0);*/
			//}
			emit = make_float4(0.1f, 0.0f, 0.3f, 0.0f); f = make_float4(0.75f, 0.75f, 0.75f, 0.0f);
			// dark ist weiß
			if (firsthit.dark == true) { emit = make_float4(0.3f, 0.3f, 0.5f, 0.0f); f = make_float4(0.0f, 1.0f, 0.0f, 0.0f); /*printf("Éncountered dark state\n");*/ }

			if (firsthit.face == 3 || firsthit.face == 6) { emit = make_float4(6, 6, 6, 0); f = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }

			if (firsthit.constrained == true) { firsthit.refl_t = DIFF; }
			if (firsthit.wall == true) { firsthit.refl_t = DIFF; }
			if (firsthit.dark == true) { firsthit.refl_t = DIFF; }
			if (edgeVisualization && isEdge) { emit = make_float4(1.0f, 1.0f, 0.0f, 0.0f); f = make_float4(1.0f, 0.0f, 0.0f, 0.0f);} // visualize wall/constrained edges
		}

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)
		accucolor += (mask * emit);

		// diffuse material, based on smallpt by Kevin Beason 
		if (firsthit.refl_t == DIFF){

			// pick two random numbers
			float phi = 2 * _PI_ * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float4 w = nl; w = normalize(w);
			float4 u = Cross((fabs(w.x) > .1 ? make_float4(0, 1, 0, 0) : make_float4(1, 0, 0, 0)), w); u = normalize(u);
			float4 v = Cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			dw = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
			dw = normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (firsthit.refl_t == METAL){

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * _PI_ * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float phongexponent = 20;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			float4 w = rayInWorldSpace - n * 2.0f * Dot(n, rayInWorldSpace); w = normalize(w);
			float4 u = Cross((fabs(w.x) > .1 ? make_float4(0, 1, 0, 0) : make_float4(1, 0, 0, 0)), w); u = normalize(u);
			float4 v = Cross(w, u); // v is normalised by default

			// compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
			dw = normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (firsthit.refl_t == SPEC){

			// compute reflected ray direction according to Snell's law
			dw = rayInWorldSpace - n * 2.0f * Dot(n, rayInWorldSpace);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01;   // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// perfectly refractive material (glass, water)
		if (firsthit.refl_t == REFR){

			bool into = Dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = Dot(rayInWorldSpace, nl);
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				dw = rayInWorldSpace;
				dw -= n * 2.0f * Dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float4 tdir = rayInWorldSpace * nnt;
				tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
				tdir = normalize(tdir);

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : Dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randState) < 0.25) // reflection ray
				{
					mask *= RP;
					dw = rayInWorldSpace;
					dw -= n * 2.0f * Dot(n, rayInWorldSpace);

					pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
				}
				else // transmission ray
				{
					mask *= TP;
					dw = tdir; //r = Ray(x, tdir); 
					pointHitInWorldSpace = x + nl * 0.001f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		originInWorldSpace = pointHitInWorldSpace;
		rayInWorldSpace = dw;
		newstart = firsthit.tet; // new tet origin
	}
	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return RGB(accucolor.x, accucolor.y, accucolor.z);
}


__global__ void renderKernel(mesh2 *tetmesh, int32_t start, float3 *accumbuffer, float3 *c, unsigned int hashedframenumber, unsigned int framenumber, float4 position, float4 view, float4 up, float fovx, float fovy, float focalDistance, float apertureRadius)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen
	float4 rendercampos = make_float4(position.x, position.y, position.z, 0);
	RGB finalcol(0);

	float distance;

	for (int s = 0; s < spp; s++)
	{
		float4 rendercamview = make_float4(view.x, view.y, view.z, 0); rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
		float4 rendercamup = make_float4(up.x, up.y, up.z, 0); rendercamup = normalize(rendercamup);
		float4 horizontalAxis = Cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
		float4 verticalAxis = Cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.
		float4 middle = rendercampos + rendercamview;
		float4 horizontal = horizontalAxis * tanf(fovx * 0.5 * (_PI_ / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		float4 vertical = verticalAxis * tanf(-fovy * 0.5 * (_PI_ / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (width - 1);
		float sy = (jitterValueY + pixely) / (height - 1);
		float4 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
		float4 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * focalDistance); // Important for depth of field!		
		float4 aperturePoint;
		if (apertureRadius > 0.00001)
		{
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);
			float angle = 2 * _PI_ * random1;
			float distance = apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;
			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { aperturePoint = rendercampos; }

		// calculate ray direction of next ray in path
		float4 apertureToImagePlane = pointOnImagePlane - aperturePoint;
		apertureToImagePlane = normalize(apertureToImagePlane); // ray direction, needs to be normalised
		float4 rayInWorldSpace = apertureToImagePlane;
		rayInWorldSpace = normalize(rayInWorldSpace);
		float4 originInWorldSpace = aperturePoint;



		finalcol += radiance(tetmesh, start, originInWorldSpace, rayInWorldSpace, rendercampos, &randState, distance) * (1.0f / spp);
	}

	accumbuffer[i] += finalcol;
	float3 tempcol = accumbuffer[i] / framenumber;

	if (distVisualization) tempcol = make_float3(0, 0, distance * 3 / 100);

	Color fcolour;
	float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / gamma) * 255), (unsigned char)(powf(colour.y, 1 / gamma) * 255), (unsigned char)(powf(colour.z, 1 / gamma) * 255), 1);
	c[i] = make_float3(x, y, fcolour.c);
}


void render()
{
	GLFWwindow* window;
	if (!glfwInit()) exit(EXIT_FAILURE);
	window = glfwCreateWindow(width, height, "", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetErrorCallback(error_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "GLEW not supported.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "GLEW successfully initialized  \n");


	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glOrtho(0.0, width, 0.0, height, 0, 1);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(float3), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsResource_t _cgr;
	size_t num_bytes;
	cudaGraphicsGLRegisterBuffer(&_cgr, vbo, cudaGraphicsRegisterFlagsNone);
	fprintf(stderr, "VBO created  \n");
	fprintf(stderr, "Entering glutMainLoop...  \n");

	my_stbtt_initfont(); // font initialization

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (bufferReset)
		{
			frameNumber = 0;
			cudaMemset(accumulatebuffer, 1, width * height * sizeof(float3));
		}
		bufferReset = false;
		frameNumber++;
		interactiveCamera->buildRenderCamera(hostRendercam);

		// Calculate deltatime of current frame
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		std::stringstream title;
		title << "Tetrahedral pathtracing with node-based tetrahedral mesh (2017) by Christian Lehmann";
		glfwSetWindowTitle(window, title.str().c_str());

		// CUDA interop
		cudaGraphicsMapResources(1, &_cgr, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&finalimage, &num_bytes, _cgr);
		glClear(GL_COLOR_BUFFER_BIT);
		dim3 block(16, 16, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		renderKernel << <grid, block >> >(mesh, _start_tet, accumulatebuffer, finalimage, WangHash(frameNumber), frameNumber,
			hostRendercam->position, hostRendercam->view, hostRendercam->up, hostRendercam->fov.x, hostRendercam->fov.x,
			hostRendercam->focalDistance, hostRendercam->apertureRadius);
		gpuErrchk(cudaDeviceSynchronize());
		cudaGraphicsUnmapResources(1, &_cgr, 0);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(2, GL_FLOAT, 12, 0);
		glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glDrawArrays(GL_POINTS, 0, width * height);
		glDisableClientState(GL_VERTEX_ARRAY);

		float mrays = width*height*MAX_DEPTH*0.000001 / deltaTime;
		std::string a = "Currently " + std::to_string(mrays) + " Mray/s";

		my_stbtt_print(100, 200, a, make_float3(1, 0, 1));
		std::string str_orig = std::to_string(hostRendercam->position.x) + " " + std::to_string(hostRendercam->position.y) + " " + std::to_string(hostRendercam->position.z);
		std::string str_dir = std::to_string(hostRendercam->view.x) + " " + std::to_string(hostRendercam->view.y) + " " + std::to_string(hostRendercam->view.z);
		my_stbtt_print(100, 150, "Ray origin: " + str_orig, make_float3(1, 0, 1));
		my_stbtt_print(100, 100, "Ray direction: " + str_dir, make_float3(1, 0, 1));
		my_stbtt_print(100, 50, "Current tet: " + std::to_string(_start_tet), make_float3(1, 0, 1));
		my_stbtt_print(100, 10, "Current ms/frame: " + std::to_string(deltaTime*1000), make_float3(1, 0, 1));

		glfwSwapBuffers(window);
	}
}


int main(int argc, char *argv[])
{
	float4 cam;
	int depth;
	int w, h;
	parseIni("test.ini", cam, depth, w, h);
	width = w;
	height = h;
	lastX = width / 2;
	lastY = height / 2;
	MAX_DEPTH = depth;

	//delete interactiveCamera;
	interactiveCamera = new InteractiveCamera();
	interactiveCamera->setResolution(width, height);
	interactiveCamera->setFOVX(45);
	hostRendercam = new Camera();
	hostRendercam->position = cam;
	interactiveCamera->buildRenderCamera(hostRendercam);

	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	// ===========================
	//     mesh2
	// ===========================

	openDialog();
	tetrahedral_mesh tetmesh;
	tetmesh.loadobj(global_filename);

	gpuErrchk(cudaMallocManaged(&mesh, sizeof(mesh2)));

	// INDICES
	mesh->oldfacenum = tetmesh.oldfacenum;
	mesh->oldnodenum = tetmesh.oldnodenum;
	mesh->facenum = tetmesh.facenum;
	mesh->nodenum = tetmesh.nodenum;
	mesh->tetnum = tetmesh.tetnum;

	// NODES - GEOMETRY MESH
	cudaMallocManaged(&mesh->ng_index, mesh->oldnodenum*sizeof(uint32_t));
	for (auto i : tetmesh.oldnodes) mesh->ng_index[i.index] = i.index;
	cudaMallocManaged(&mesh->ng_x, mesh->oldnodenum*sizeof(float));
	cudaMallocManaged(&mesh->ng_y, mesh->oldnodenum*sizeof(float));
	cudaMallocManaged(&mesh->ng_z, mesh->oldnodenum*sizeof(float));
	for (auto i : tetmesh.oldnodes) mesh->ng_x[i.index] = i.x;
	for (auto i : tetmesh.oldnodes) mesh->ng_y[i.index] = i.y;
	for (auto i : tetmesh.oldnodes) mesh->ng_z[i.index] = i.z;

	// FACES - GEOMETRY MESH
	cudaMallocManaged(&mesh->fg_index, mesh->oldfacenum*sizeof(uint32_t));
	for (auto i : tetmesh.oldfaces) mesh->fg_index[i.index] = i.index;
	cudaMallocManaged(&mesh->fg_node_a, mesh->oldfacenum*sizeof(uint32_t));
	cudaMallocManaged(&mesh->fg_node_b, mesh->oldfacenum*sizeof(uint32_t));
	cudaMallocManaged(&mesh->fg_node_c, mesh->oldfacenum*sizeof(uint32_t));
	for (auto i : tetmesh.oldfaces) mesh->fg_node_a[i.index] = i.node_a;
	for (auto i : tetmesh.oldfaces) mesh->fg_node_b[i.index] = i.node_b;
	for (auto i : tetmesh.oldfaces) mesh->fg_node_c[i.index] = i.node_c;

	// NODES
	cudaMallocManaged(&mesh->n_index, mesh->nodenum*sizeof(uint32_t));
	for (auto i : tetmesh.nodes) mesh->n_index[i.index] = i.index;
	cudaMallocManaged(&mesh->n_x, mesh->nodenum*sizeof(float));
	cudaMallocManaged(&mesh->n_y, mesh->nodenum*sizeof(float));
	cudaMallocManaged(&mesh->n_z, mesh->nodenum*sizeof(float));
	for (auto i : tetmesh.nodes) mesh->n_x[i.index] = i.x;
	for (auto i : tetmesh.nodes) mesh->n_y[i.index] = i.y;
	for (auto i : tetmesh.nodes) mesh->n_z[i.index] = i.z;

	// ASSIGN FACES
	gpuErrchk(cudaMallocManaged(&mesh->assgndata, tetmesh.tetnum*99*sizeof(int32_t))); // 6 tets * 99 faces per tet
	for (size_t i = 0; i < tetmesh.tetnum * 99; i++){ mesh->assgndata[i] = -1; } // alle auf -1 setzen

	for (int i = 0; i < tetmesh.tetrahedras.size();i++) // loop over all tetrahedra
	{
		for (int j = 0; j < 99; j++) // loop over all faces per tet
		{
			if (j < tetmesh.tetrahedras.at(i).counter) { if (tetmesh.tetrahedras.at(i).faces[j] >= 0) mesh->assgndata[i * 99 + j] = tetmesh.tetrahedras.at(i).faces[j]; }
			// else mesh->assgndata[i * 99 + j] = -1;
		}
	}

	// TETRAHEDRA
	cudaMallocManaged(&mesh->t_index, mesh->tetnum*sizeof(uint32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_index[i.number] = i.number;
	cudaMallocManaged(&mesh->t_findex1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_findex1[i.number] = i.findex1;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex2[i.number] = i.findex2;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex3[i.number] = i.findex3;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex4[i.number] = i.findex4;
	cudaMallocManaged(&mesh->t_nindex1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex1[i.number] = i.nindex1;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex2[i.number] = i.nindex2;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex3[i.number] = i.nindex3;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex4[i.number] = i.nindex4;
	cudaMallocManaged(&mesh->t_adjtet1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet1[i.number] = i.adjtet1;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet2[i.number] = i.adjtet2;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet3[i.number] = i.adjtet3;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet4[i.number] = i.adjtet4;

	// ===========================
	//     mesh end
	// ===========================

	 
	// Get bounding box
	box = init_BBox(mesh);
	fprintf_s(stderr, "\nBounding box:MIN xyz - %f %f %f \n", box.min.x, box.min.y, box.min.z);
	fprintf_s(stderr, "             MAX xyz - %f %f %f \n\n", box.max.x, box.max.y, box.max.z);

	// Allocate unified memory
	gpuErrchk(cudaMallocManaged(&finalimage, width * height * sizeof(float3)));
	gpuErrchk(cudaMallocManaged(&accumulatebuffer, width * height * sizeof(float3)));

	// find starting tetrahedra
	uint32_t _dim = 2 + pow(mesh->tetnum, 0.25);
	dim3 Block(_dim, _dim, 1);
	dim3 Grid(_dim, _dim, 1);
	//uint32_t _dim = mesh->tetnum;
	//dim3 Block(_dim, _dim, 1);
	//dim3 Grid(_dim, _dim, 1);
	GetTetrahedraFromPoint << <mesh->tetnum, 1 >> >(mesh, hostRendercam->position);
	gpuErrchk(cudaDeviceSynchronize());





	fprintf(stderr, "Starting point coordinates: %f %f %f \n",hostRendercam->position.x, hostRendercam->position.y, hostRendercam->position.z);

	if (_start_tet == -1) 
	{
		fprintf(stderr, "Starting point outside tetrahedra! Aborting ... \n");
		system("PAUSE");
		exit(0);

	}
	else fprintf(stderr, "Starting tetrahedra - camera: %lu \n", _start_tet);

	// main render function
	render();

	gpuErrchk(cudaDeviceReset());
	glfwTerminate();
	return 0;
}

