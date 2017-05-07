/*
*  tetrahedra-based raytracer
*  Copyright (C) 2015  Christian Lehmann
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

#pragma once
#include <string>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

typedef int int32_t;
typedef unsigned int uint32_t;

#define _PI_ 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define PI_OVER_TWO 1.5707963267948966192313216916397514420985f
#define eps 1e-8
#define inf 1e20

enum Refl_t { DIFF, SPEC, REFR, VOL, METAL}; 
enum Geometry { TRIANGLE, SPHERE };

// ----------------  CUDA float operations -------------------------------

inline __host__ __device__ float3 operator/(const float3 &a, const int &b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ float3 operator+=(float3 &a, const float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return make_float3(0, 0, 0); }

inline __host__ __device__ float4 operator-=(float4 &a, const float4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;	return make_float4(0, 0, 0, 0); }
inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0); }
inline __host__ __device__ float4 operator-(const float4 &a, const float4 &b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0); }
inline __host__ __device__ float4 operator*(float4 &a, float4 &b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, 0); }
inline __host__ __device__ float4 operator*(const float4 &a, const float &b) { return make_float4(a.x*b, a.y*b, a.z*b, 0); }
inline __host__ __device__ float4 operator*(const float &b, const float4 &a) { return make_float4(a.x * b, a.y * b, a.z * b, 0); }
inline __host__ __device__ void operator*=(float4 &a, float4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
inline __host__ __device__ void operator*=(float4 &a, float &b) { a.x *= b; a.y *= b; a.z *= b; }
inline __host__ __device__ float4 operator/(const float4 &a, const float &b) { return make_float4(a.x / b, a.y / b, a.z / b, 0); }
inline __host__ __device__ float4 operator+=(float4 &a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; return make_float4(0, 0, 0, 0); }

// ------------------------CUDA math --------------------------------------------------

__device__ float4 normalize(float4 &a)
{ 
	float f = 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); 
	return make_float4(a.x*f, a.y*f, a.z*f, 0);
}

 __device__  __host__ float Dot(const float4 &a, const float4 &b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float4 reflect(const float4 &i,const float4 &n)
{
	return i - 2.0f * n * Dot(n, i);
}

__device__  float4 Cross(const float4 &a, const float4 &b)
{
	return make_float4( a.y * b.z - a.z * b.y, 
						a.z * b.x - a.x * b.z, 
						a.x * b.y - a.y * b.x, 0);
}

__device__ float ScTP(const float4 &a, const float4 &b, const float4 &c)
{
	// computes scalar triple product
	return Dot(a, Cross(b, c));
}

__device__ int signf(float f) 
{
	if (f > 0.0) return 1;
	if (f < 0.0) return -1;
	return 0;
}

__device__ bool sameSign(float a, float b)
{
	if (signf(a) == signf(b)) return true;
	return false;
}

__device__ bool SameSide(const float4 &v1, const float4 &v2, const float4 &v3, const float4 &v4, const float4 &p)
{
	float4 normal = Cross(v2 - v1, v3 - v1);
	float dotV4 = Dot(normal, v4 - v1);
	float dotP = Dot(normal, p - v1);
	return signf(dotV4) == signf(dotP);
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}

__device__ bool HasNaNs(float4 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z); }

__device__ float4 mix(float4 x, float4 y, float a) { return(x*(-a + 1) + y*a); }

struct RGB
{
	float x, y, z;
	__device__ RGB(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	__device__ RGB(float xyz0){ x = xyz0; y = xyz0; z = xyz0; }
	__device__ RGB operator/(const float &b) const { return RGB(x / b, y / b, z / b); }
	__device__ RGB operator+(const RGB &b) const { return RGB(x + b.x, y + b.y, z + b.z); }
	__device__ RGB operator*(const float &b) const { return RGB(x * b, y * b, z * b); }
};
__device__ RGB operator+=(RGB &a, const RGB b) { a.x += b.x; a.y += b.y; a.z += b.z; return RGB(a.x,a.y,a.z); }
__device__ float3 operator+=(float3 &a, const RGB b) { a = make_float3(a.x + b.x, a.y + b.y, a.z + b.z); return a; }

__device__ RGB de_nan(RGB &a)
{
	// from http://psgraphics.blogspot.de/2016/04/debugging-by-sweeping-under-rug.html
	RGB temp = a;
	if (!(temp.x == temp.x)) temp.x = 0;
	if (!(temp.y == temp.y)) temp.y = 0;
	if (!(temp.z == temp.z)) temp.z = 0;
	temp.z = 0;
	return temp;
}

struct Ray
{
	float4 o, d;
	__device__ __host__ Ray(float4 o_, float4 d_) : o(o_), d(d_) {}
};

__device__ bool nearlyzero(float a)
{
	if (a > 0.0 && a < 0.0001) return true;
	return false;
}

__device__ float dist(float4 a, float4 b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

// ----------------------- non-CUDA math -----------------------

inline float4 minCPU(const float4& a, const float4& b) 
{
	return make_float4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w));
}

inline float4 maxCPU(const float4& a, const float4& b) 
{
	return make_float4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w));
}

float4 normalizeCPU(const float4 &a)
{
	float f = 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
	return make_float4(a.x * f, a.y * f, a.z * f, 0);
}

float4 CrossCPU(const float4 &a, const float4 &b)
{
	float4 cross = { a.y * b.z - a.z * b.y,
					a.z * b.x - a.x * b.z,
					a.x * b.y - a.y * b.x };
	return cross;
}

float DotCPU(const float4 &a, const float4 &b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

int signfCPU(float f) 
{
	if (f > 0.0) return 1;
	if (f < 0.0) return -1;
	return 0;
}

bool sameSignCPU(float a, float b)
{
	if (signfCPU(a) == signfCPU(b)) return true;
	return false;
}

bool SameSideCPU(const float4 &v1, const float4 &v2, const float4 &v3, const float4 &v4, const float4 &p)
{
	float4 normal = CrossCPU(v2 - v1, v3 - v1);
	float dotV4 = DotCPU(normal, v4 - v1);
	float dotP = DotCPU(normal, p - v1);
	return signfCPU(dotV4) == signfCPU(dotP);
}
 
float ScTPCPU(const float4 &a, const float4 &b, const float4 &c)
{
	// computes scalar triple product
	return DotCPU(a, CrossCPU(b, c));
}

bool nearlysame(float a, float b)
{
	float _eps = 0.01;
	if (abs(a - b) < _eps ) return true;
	return false;
}

bool RayTriangleIntersectionCPU(const float4&p, const float4& q,	const float4 &v0,	const float4 &v1,	const float4 &v2)
{
/* // ray-triangle
float u, v, w;
float4 pq = q - p; 
float4 pa = v0 - p; 
float4 pb = v1 - p; 
float4 pc = v2 - p; 

float4 m = CrossCPU(pq, pc); 
u = DotCPU(pb, m); 
v = -DotCPU(pa, m); 
if (!sameSignCPU(u, v)) return false; 
w = ScTPCPU(pq, pb, pa); 
if (!sameSignCPU(u, w)) return false;
return true;*/

	/* // segment-triangle
	float t,u,v,w;
	float4 ab = v1 - v0; 
	float4 ac = v2 - v0; 
	float4 qp = p - q;
	float4 n = CrossCPU(ab, ac);
	float d = DotCPU(qp, n); 
	if (d <= 0.0f) return false;
	float4 ap = p - v0; 
	t = DotCPU(ap, n); 
	if (t < 0.0f) return false; 
	if (t > d) return false; 
	float4 e = CrossCPU(qp, ap); 
	v = DotCPU(ac, e); 
	if (v < 0.0f || v > d) return false; 
	w = -DotCPU(ab, e); 
	if (w < 0.0f || v + w > d) return false;
	return true;*/

	// Chirkov2005 - line segment-triangle
	// https://github.com/erich666/jgt-code/blob/master/Volume_10/Number_3/Chirkov2005/src/C2005.cpp
	float4 org = p;
	float4 end = q;
	float4 e0 = v1 - v0;
	float4 e1 = v2 - v0;
	float4 norm = CrossCPU(e0,e1);
	float pd = DotCPU(norm, v0);
	float signSrc = DotCPU(norm, org) - pd;
	float signDst = DotCPU(norm, end) - pd;
	if(signSrc*signDst > 0.0) return false;
	float d = signSrc/(signSrc - signDst);
	float4 point = org + d*(end - org);
	float4 v = point - v0;		
	float4 av = CrossCPU(e0,v);
	float4 vb = CrossCPU(v,e1);
	if(DotCPU(av,vb) > 0.0)
	{
		float4 e2 = v1 - v2;
		float4 v = point - v1;
		float4 vc = CrossCPU(v,e2);
		if(DotCPU(av,vc) > 0.0) return true;
	}
	return false;

}


bool RayTetIntersectionCPU(const float4 &p1, const float4 &p2, const float4 &v0, const float4 &v1, const float4 &v2, const float4 &v3)
{
	if (RayTriangleIntersectionCPU(p1, p2 , v0, v1, v2)) return true;
	if (RayTriangleIntersectionCPU(p1, p2 , v0, v2, v3)) return true;
	if (RayTriangleIntersectionCPU(p1, p2 , v1, v2, v3)) return true;
	if (RayTriangleIntersectionCPU(p1, p2 , v0, v1, v3)) return true;
	return false;
}


// ------------------------------- structure definitions -----------------------------
 
struct BBox
{
	float4 min, max;
};

void scale_BBox(BBox &box, const float& factor)
{
box.min = box.min * factor;
box.max = box.max * factor;
}

struct mesh2
{
	// nodes - geometry mesh
	uint32_t *ng_index;
	float *ng_x, *ng_y, *ng_z;

	//faces - geometry mesh
	uint32_t *fg_index;
	uint32_t *fg_node_a, *fg_node_b, *fg_node_c;

	// nodes
	uint32_t *n_index;
	float *n_x, *n_y, *n_z;

	// tetrahedra
	uint32_t *t_index;
	int32_t *t_findex1, *t_findex2, *t_findex3, *t_findex4;
	int32_t *t_nindex1, *t_nindex2, *t_nindex3, *t_nindex4;
	int32_t *t_adjtet1, *t_adjtet2, *t_adjtet3, *t_adjtet4;

	//mesh 
	uint32_t tetnum, nodenum, facenum, oldnodenum, oldfacenum;

	// array for assigning faces to tets
	int32_t *assgndata;
};

struct rayhit
{
	float4 pos;
	float4 color;
	float4 ref;
	Refl_t refl_t;
	int32_t tet = 0;
	int32_t face = 0;
	int depth = 0;
	bool wall = false;
	bool constrained = false;
	bool dark = false; // if hit is too far away
};

