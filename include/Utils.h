#pragma once
#include <string>
#include <stdio.h>

int ReadLowEndian32(int low) {
	return (low << 24) & 0xFF000000 | (low << 8) & 0x00FF0000 | (low >> 24) & 0x000000FF | (low >> 8) & 0x0000FF00;
}



struct GrayScaleBitmap {

	int width;
	int height;
	int size;
	int headerSize;

	char header[122];
	unsigned int* data;


	GrayScaleBitmap(int width, int height) :width(width), height(height) {
		memset(header, 0, 122);
		header[0] = 'B';
		header[1] = 'M';
		*(unsigned int*)(header + 2) = width * height;
		header[6] = 0;
		header[7] = 0;
		header[8] = 0;
		header[9] = 0;
		*(unsigned int*)(header + 10) = 122;
		*(unsigned int*)(header + 14) = 108;
		*(unsigned int*)(header + 18) = width;
		*(unsigned int*)(header + 22) = height;
		*(unsigned short*)(header + 26) = 1;
		*(unsigned short*)(header + 28) = 32;
		*(unsigned int*)(header + 0x26) = 2835;
		*(unsigned int*)(header + 0x2A) = 2835;
		*(unsigned char*)(header + 0x46) = 'W';
		*(unsigned char*)(header + 0x47) = 'i';
		*(unsigned char*)(header + 0x48) = 'n';
		*(unsigned char*)(header + 0x49) = ' ';

		data = (unsigned int*)malloc(width * height * sizeof(int));
		size = width * height;
	}

	~GrayScaleBitmap() { free(data); }
};

void SaveGSBitmap(std::string file, GrayScaleBitmap& img) {
	FILE* out = fopen(file.c_str(), "wb");
	fwrite(img.header, 1, 122, out);
	for(int i = img.height-1; i >= 0; i--) {
		fwrite(img.data + i * img.width, 4, img.width, out);
	}
	fclose(out);
}


//Byte goes from 0-255
unsigned int GrayScaleValue(unsigned char byte) {
	int temp = 0;
	int val = byte;
	return temp | (val << 16) | (val << 8) | val;
}

unsigned int GrayScaleValue(float intensity) {
	int temp = 0;
	if (intensity < 0)intensity = 0;
	int val = intensity*255;
	return temp | (val << 16) | (val << 8) | val;
}