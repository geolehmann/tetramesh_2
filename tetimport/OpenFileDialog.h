#pragma once
#include <Windows.h>
#include <Commdlg.h>
#include <string>
#include <algorithm>

std::string global_filename;

OPENFILENAME    ofn;
char *FilterSpec = "GOCAD ASCII Files(*.*)\0*.*\0"; //Object Files(*.obj)\0*.obj\0Text Files(*.txt)\0*.txt\0All Files
char *Title = "Load GOCAD ASCII File";
char szFileName[MAX_PATH];
char szFileTitle[MAX_PATH];
int             Result;

int openDialog()
{

	/* fill in non-variant fields of OPENFILENAME struct. */
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = GetFocus();
	ofn.lpstrFilter = FilterSpec;
	ofn.lpstrCustomFilter = NULL;
	ofn.nMaxCustFilter = 0;
	ofn.nFilterIndex = 0;
	ofn.lpstrFile = szFileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrInitialDir = "."; // Initial directory.
	ofn.lpstrFileTitle = szFileTitle;
	ofn.nMaxFileTitle = MAX_PATH;
	ofn.lpstrTitle = Title;
	ofn.lpstrDefExt = NULL;

	ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;

	if (!GetOpenFileName((LPOPENFILENAME)&ofn))
	{
		return -1; // Failed or cancelled
	}
	else
	{
		global_filename = szFileName;
	}
	return 0;
}

int parseIni(std::string filename, float4 &campos, int &mdepth, int& x, int& y)
{
	std::ifstream ifs(filename.c_str(), std::ifstream::in);
	if (!ifs.good())
	{
		std::cout << "Error loading config:(" << filename << ") file not found!" << "\n";
		system("PAUSE");
		exit(0);
	}

	std::string line, key;
	int vertexid = 0, faceid = 0;
	std::cout << "Started loading config file " << filename <<". \n";
	int linecounter = 0;
	while (!ifs.eof() && std::getline(ifs, line))
	{
		key = "";
		std::stringstream stringstream(line);
		stringstream >> key >> std::ws;

		if (key == "orig") 
		{
			float x, y, z;
			stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;	
			campos = make_float4(x, y, z, 0);
		}

		if (key == "depth") 
		{
			int x;
			stringstream >> x >> std::ws;	
			mdepth = x;
		}

		if (key == "res") 
		{
			int x_, y_;
			stringstream >> x_ >> std::ws >> y_ >> std::ws;	
			x = x_;
			y = y_;
		}
	}
}