#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sound_classification_v2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sound_classification, m) {
  py::class_<SoundClassificationV2>(m, "SoundClassificationV2")
      .def(py::init<int, std::string, float>(), py::arg("dev_id"),
           py::arg("bmodel_path"), py::arg("threshold"))
      .def(
          "inference",
          [](SoundClassificationV2& self, py::array_t<short> temp_buffer) {
            // Ensure the buffer is contiguous
            py::buffer_info buf_info = temp_buffer.request();
            // if (buf_info.ndim != 1) {
            //   throw std::runtime_error("Input array must be 1-dimensional");
            // }
            if (buf_info.itemsize != sizeof(short)) {
              throw std::runtime_error("Input array must be of type int16");
            }
            int buffer_size = buf_info.shape[0];
            short* ptr = static_cast<short*>(buf_info.ptr);
            return self.inference(ptr, buffer_size);
          },
          py::arg("temp_buffer"))

      .def("get_status", &SoundClassificationV2::get_status)
      .def("get_threshold", &SoundClassificationV2::get_threshold)
      .def("set_threshold", &SoundClassificationV2::set_threshold,
           py::arg("threshold"))
      .def("set_logger_level", &SoundClassificationV2::set_logger_level,
            py::arg("level"));
}