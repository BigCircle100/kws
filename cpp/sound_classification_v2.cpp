#include "sound_classification_v2.hpp"
#include <iostream>
#include <numeric>
#include <cassert>
#include <fstream>
#include <spdlog/spdlog.h>
using namespace melspec;
using namespace std;

SoundClassificationV2::SoundClassificationV2(int dev_id, 
                                            std::string bmodel_path, 
                                            float threshold) : m_dev_id(dev_id), m_model_threshold(threshold), m_status(0) {
  // logger
  m_logger = spdlog::stdout_color_mt("kws logger");
  m_logger->set_level(spdlog::level::info);
  
  // preprocess parameters
  audio_param_.win_len = 1024;
  audio_param_.num_fft = 1024;
  audio_param_.hop_len = 128;     
  audio_param_.sample_rate = 8000;
  audio_param_.time_len = 2;  
  audio_param_.num_mel = 40;
  audio_param_.fmin = 0;
  audio_param_.fmax = audio_param_.sample_rate / 2;
  audio_param_.fix = true;        // 已按照要求修改

  bm_status_t ret_bm = BM_SUCCESS;
  bool ret = true;

  // runtime
  ret_bm = bm_dev_request(&m_handle, dev_id);
  if (ret_bm != BM_SUCCESS){
    m_logger->error("get handle failed with dev_id {}", dev_id);
    throw runtime_error("get handle failed");
  }
  
  m_bmrt = bmrt_create(m_handle);
  ret = bmrt_load_bmodel(m_bmrt, bmodel_path.c_str());
  if (!ret){
    m_logger->error("load bmodel failed with bmodel_path: {}", bmodel_path);
    throw runtime_error("load bmodel failed");
  }

  bmrt_get_network_names(m_bmrt, &m_net_names);
  m_net_info = bmrt_get_network_info(m_bmrt, m_net_names[0]);

// 目前不确定模型的输入输出tensor个数，后续拿到bmodel再改

  m_input_num = m_input_num = m_net_info->input_num;
  m_input_shapes.emplace_back(m_net_info->stages[0].input_shapes[0]);
  m_input_scale = m_net_info->input_scales[0];

  m_output_num = m_output_num = m_net_info->output_num;
  m_output_shapes.emplace_back(m_net_info->stages[0].output_shapes[0]);
  m_output_scale = m_net_info->output_scales[0];
  
  // 根据input和output tensor大小来计算对应的系统内存大小，这里假设输入输出tensor是float
  int input_shape_size = 1;
  for (int i = 0; i < m_input_shapes[0].num_dims; ++i){
    input_shape_size *= m_input_shapes[0].dims[i];
  }
  h_input_size = input_shape_size*sizeof(float);
  
  m_output_element = 1;
  for (int i = 0; i < m_output_shapes[0].num_dims; ++i){
    m_output_element *= m_output_shapes[0].dims[i];
  }
  h_output_size = m_output_element*sizeof(float);

  h_input_buffer = malloc(h_input_size);
  h_output_buffer = malloc(h_output_size);

  m_input_buffers.emplace_back(h_input_buffer);
  m_output_buffers.emplace_back(h_output_buffer);

  onModelOpened();

  m_status = 1;

  m_logger->info("init finished");
}

SoundClassificationV2::~SoundClassificationV2() { 
  delete mp_extractor_; 
  bmrt_destroy(m_bmrt);
  bm_dev_free(m_handle);  
  for(auto &buffer: m_input_buffers){
    free(buffer);
  }
  for(auto &buffer: m_output_buffers){
    free(buffer);
  }

}

int SoundClassificationV2::onModelOpened() {

  bool htk = false;

  audio_param_.fmax = audio_param_.sample_rate / 2;
  int num_frames = audio_param_.time_len * audio_param_.sample_rate;

  mp_extractor_ = new MelFeatureExtract(num_frames, audio_param_.sample_rate, audio_param_.num_fft,
                                        audio_param_.hop_len, audio_param_.num_mel,
                                        audio_param_.fmin, audio_param_.fmax, "reflect", htk);
  return 0;
}

AudioAlgParam SoundClassificationV2::get_algparam() { return audio_param_; }
void SoundClassificationV2::set_algparam(AudioAlgParam audio_param) {
  audio_param_.win_len = audio_param.win_len;
  audio_param_.num_fft = audio_param.num_fft;
  audio_param_.hop_len = audio_param.hop_len;
  audio_param_.sample_rate = audio_param.sample_rate;
  audio_param_.time_len = audio_param.time_len;
  audio_param_.num_mel = audio_param.num_mel;
  audio_param_.fmin = audio_param.fmin;
  audio_param_.fmax = audio_param.fmax;
  audio_param_.fix = audio_param.fix;
}

std::pair<int, float> SoundClassificationV2::inference(short *temp_buffer, int buffer_size) {

  m_status = 2;

  normal_sound(temp_buffer, buffer_size);
  mp_extractor_->update_data(temp_buffer, buffer_size);

  mp_extractor_->melspectrogram_optimze(temp_buffer, buffer_size, static_cast<float*>(m_input_buffers[0]),
                                        h_input_size, m_input_scale, audio_param_.fix);

  bool ret_rt = bmrt_launch_data(m_bmrt, 
                                m_net_names[0], 
                                m_input_buffers.data(), 
                                m_input_shapes.data(), 
                                m_input_num, 
                                m_output_buffers.data(), 
                                m_output_shapes.data(),
                                m_output_num,
                                true);
  if (!ret_rt){
    m_logger->error("inference failed");
    throw runtime_error("inference failed");
  }

  
  auto result = get_top_k(static_cast<float*>(m_output_buffers[0]), m_output_element);

  m_status = 3;

  return result;
}


std::pair<int, float> SoundClassificationV2::get_top_k(float *result, size_t count) {
  int idx = -1;
  float max_e = -10000;
  float cur_e;

  float sum_e = 0.;
  for (size_t i = 0; i < count; i++) {
    cur_e = std::exp(result[i]);
    if (i != 0 && cur_e > max_e) {
      max_e = cur_e;
      idx = i;
    }
    sum_e = float(sum_e) + float(cur_e);
    // std::cout << i << ": " << cur_e << "\t";
  }
  // std::cout << "\n";
  // for (size_t i = 0; i < count; i++) {
  //   cur_e = std::exp(result[i]) / sum_e;
  //   std::cout << "  i:" << i << ", score:" << cur_e;
  // }

  float max = max_e / sum_e;
  if (idx != 0 && max < m_model_threshold) {
    idx = 0;
  }

// std::cout << "all probs: ";
//   for (int i = 0; i < count; ++i){
//     std::cout << std::exp(result[i])/sum_e << " ";
//   }
//   std::cout << std::endl;
  return {idx, max};
}

void SoundClassificationV2::normal_sound(short *audio_data, int n) {
  // std::cout << "before:" << audio_data[0];
  std::vector<double> audio_abs(n);
  for (int i = 0; i < n; i++) {
    audio_abs[i] = std::abs(static_cast<double>(audio_data[i]));
  }
  std::vector<double> top_data;
  std::make_heap(audio_abs.begin(), audio_abs.end());
  if (top_num <= 0) {
    std::cerr << "When top_num<=0, the volume adaptive algorithm will fail. Current top_num="
              << top_num << std::endl;
    return;
  }
  for (int i = 0; i < top_num; i++) {
    top_data.push_back(audio_abs.front());
    std::pop_heap(audio_abs.begin(), audio_abs.end());
    audio_abs.pop_back();
  }
  double top_mean = std::accumulate(top_data.begin(), top_data.end(), 0.0) / top_num;
  if (top_mean == 0) {
    std::cout << "The average of the top data is zero, cannot scale the audio data." << std::endl;
  } else {
    double r = max_rate * SCALE_FACTOR_FOR_INT16 / double(top_mean);
    double tmp = 0;
    for (int i = 0; i < n; i++) {
      tmp = audio_data[i] * r;
      audio_data[i] = short(tmp);
    }
  }
  // std::cout << ", after:" << audio_data[0];
}

// server
int SoundClassificationV2::get_status(){
  return m_status;
}

float SoundClassificationV2::get_threshold(){
  return m_model_threshold;
}

void SoundClassificationV2::set_threshold(float threshold){
  m_model_threshold = threshold;
}


void SoundClassificationV2::set_logger_level(string level){
  if (level == "INFO"){
    m_logger->set_level(spdlog::level::info);
  }else if(level == "WARNING"){
    m_logger->set_level(spdlog::level::warn);
  }else if(level == "ERROR"){
    m_logger->set_level(spdlog::level::err);
  }else if(level == "DEBUG"){
    m_logger->set_level(spdlog::level::debug);
  }else{
    m_logger->set_level(spdlog::level::info);
    m_logger->info("cannot find log level {}, using level *info* as default", level);
  }
}