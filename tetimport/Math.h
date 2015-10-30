#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stdint.h>

 struct uVec2
{
	int32_t face, tet;
	uVec2(int32_t f0, int32_t t0) { face = f0; tet = t0; }
	uVec2(int32_t x = 0) { face = 0; tet = 0; }
};
 
struct Vec {
	double x, y, z;
	Vec(double x0, double y0, double z0){ x = x0; y = y0; z = z0; }
	Vec(double xyz0 = 0){ x = xyz0; y = xyz0; z = xyz0; }
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator+=(const Vec &b) { x += b.x; y += b.y; z += b.z; return (*this); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
	Vec operator*(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	Vec operator*=(const Vec &b) { x *= b.x; y *= b.y; z *= b.z; return (*this); }
	Vec operator/(double b) const { return Vec(x / b, y / b, z / b); }
	Vec operator/(const Vec &b) const { return Vec(x / b.x, y / b.y, z / b.z); }
	bool operator<(const Vec &b) const { return x < b.x && y < b.y && z < b.z; }
	bool operator>(const Vec &b) const { return x > b.x && y > b.y && z > b.z; }
	Vec& norm(){ return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
	double length() const { return sqrt(x*x + y*y + z*z); }
	double dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; }
	double avg() const { return (x + y + z) / 3.0; }
	double max() const { return x > y ? (x > z ? x : z) : (y > z ? y : z); }
	double min() const { return x < y ? (x < z ? x : z) : (y < z ? y : z); }
	Vec operator%(const Vec &b) const { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
	const double& operator[](size_t i) const { return i == 0 ? x : (i == 1 ? y : z); }
};


float4 operator-(const float4 &a, const float4 &b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);

}


inline float Dot(const float4 a, const float4 b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float4 Cross(const float4 a, const float4 b)
{
	float4 cross = { a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x };
	return cross;
}

struct Ray 
{ 
	float4 o, d; 
};

float ScTP(const float4 a, const float4 b, const float4 c)
{
	return Dot(a, Cross(b, c));
}

int signf(float x)
{
	if (x > 0.f) return 1;
	if (x < 0.f) return -1;
	return 0;
}


bool SameSide(float4 v1, float4 v2, float4 v3, float4 v4, float4 p)
{
	float4 normal = Cross(v2 - v1, v3 - v1);
	float dotV4 = Dot(normal, v4 - v1);
	float dotP = Dot(normal, p - v1);
	return signf(dotV4) == signf(dotP);
}


struct BBox
{
	float4 min, max;
};

struct int4
{
	int32_t x, y, z, w;
};