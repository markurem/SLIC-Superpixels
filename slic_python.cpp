#include <iostream>
#include <tuple>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/range/irange.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "slic.h"

// This method copies the image content a few times, so not really efficient. (but simplest)
IplImage* createIplImageFromPythonNDarray(boost::python::object& image) {
	// Unpack data.
	const std::string data = boost::python::extract<std::string>(image.attr("tostring")());

	// Check if supported.
	const int height = boost::python::extract<int>(image.attr("shape")[0]);
	const int width = boost::python::extract<int>(image.attr("shape")[1]);
	const int channels = boost::python::extract<int>(image.attr("shape")[2]);
	assert(data.size() == width * height * channels);

	// Create and fill data.
	IplImage* result = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channels);
	for(int y : boost::irange(0, height)) {
		for(int x : boost::irange(0, width)) {
			const int offset = (y * width + x) * 3;
			cvSet2D(result, y, x, CV_RGB(
				(uint8_t)data[offset + 2],
				(uint8_t)data[offset + 1],
				(uint8_t)data[offset + 0]));
		}
	}
	return result;
}

boost::python::object createPythonNDArrayFromIplImage(const IplImage* image) {
	std::string data(image->height * image->width * 3, '\0');
	for(int y : boost::irange(0, image->height)) {
		for(int x : boost::irange(0, image->width)) {
			const CvScalar value = cvGet2D(image, y, x);
			const int offset = (y * image->width + x) * 3;
			data[offset + 0] = value.val[0];
			data[offset + 1] = value.val[1];
			data[offset + 2] = value.val[2];
		}
	}

	boost::python::list shape;
	shape.append(image->height);
	shape.append(image->width);
	shape.append(3);

	boost::python::object numpy = boost::python::import("numpy");
	return numpy
		.attr("fromstring")(data, "uint8")
		.attr("reshape")(shape);
}

boost::python::object oversegmentate(boost::python::object input, int num_superpixel, int regularity) {
	IplImage* image_rgb = createIplImageFromPythonNDarray(input);
	IplImage* image_lab = cvCloneImage(image_rgb);
	cvCvtColor(image_rgb, image_lab, CV_BGR2Lab);

	Slic slic;
	const double step = std::sqrt((image_lab->width * image_lab->height) / (double) num_superpixel);
	slic.generate_superpixels(image_lab, step, regularity);
	slic.create_connectivity(image_lab);
	slic.display_contours(image_rgb, CV_RGB(255, 0, 0));
	boost::python::object result = createPythonNDArrayFromIplImage(image_rgb);
	cvReleaseImage(&image_rgb);
	cvReleaseImage(&image_lab);

	return result;
}

BOOST_PYTHON_MODULE(slic) {
	boost::python::def("oversegmentate", &oversegmentate,
		("image", boost::python::arg("num_superpixel")=200, boost::python::arg("regularity")=80),
		"Calculate superpixels of a RGB image.\n"
		"* image: numpy array representing RGB image\n"
		"* num_superpixel: number of superpixels\n"
		"* regularity: grid-likeness of the superpixels");
}
