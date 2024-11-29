#pragma once

#include "melspec.hpp"
#include "bmruntime_interface.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#define SCALE_FACTOR_FOR_INT16 32768.0

typedef struct {
  int win_len;
  int num_fft;
  int hop_len;
  int sample_rate;
  int time_len;
  int num_mel;
  int fmin;
  int fmax;
  bool fix;
} AudioAlgParam;


class SoundClassificationV2 {
 public:
  SoundClassificationV2(int dev_id, std::string bmodel_path, float threshold);
  virtual ~SoundClassificationV2();
  int onModelOpened();
  std::pair<int, float> inference(short *temp_buffer, int buffer_size);

  std::pair<int, float> get_top_k(float *result, size_t count);
  void normal_sound(short *temp_buffer, int n);
  AudioAlgParam get_algparam();
  void set_algparam(AudioAlgParam audio_param);

// server
  int get_status();
  float get_threshold();
  void set_threshold(float threshold);
  void set_logger_level(std::string level);

 private:
// server
  int m_status;  // 0 初始化中， 1 初始化完成，2 推理中，3 推理完成

// preprocess
  float threshold_;
  melspec::MelFeatureExtract *mp_extractor_ = nullptr;
  int top_num = 500;
  float max_rate = 0.2;
  AudioAlgParam audio_param_;

// runtime
  int m_dev_id;
  bm_handle_t m_handle;
  void *m_bmrt;
  const char **m_net_names = NULL;
  const bm_net_info_t* m_net_info;

  int m_input_num;
  float m_input_scale;
  bm_data_type_t m_input_dtype;
  std::vector<bm_shape_t> m_input_shapes;
  std::vector<void *> m_input_buffers;


  int m_output_num;
  float m_output_scale;
  bm_data_type_t m_output_dtype;
  std::vector<bm_shape_t> m_output_shapes;
  int m_output_element;
  std::vector<void *> m_output_buffers;

  int h_input_size;
  int h_output_size;
  void * h_input_buffer;
  void * h_output_buffer;

  // postprocess
  float m_model_threshold;

// logger
  std::shared_ptr<spdlog::logger> m_logger;
};

