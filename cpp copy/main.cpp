#include <iostream>
#include <fstream>
#include "sound_classification_v2.hpp"
#define AUDIOFORMATSIZE 2


std::vector<std::string> classname = {"背景","你好算能","清除缓存","清空缓存"};

int main(int argc, char* argv[]){

  // args: input.bin bmodel_path
  int dev_id = 0;
  std::string filename = argv[1];
  std::string bmodel_path = argv[2];
  
  float threshold = 0.5;
  int sample_rate = 8000;
  int seconds = 2;
  int frame_size = sample_rate * AUDIOFORMATSIZE * seconds;

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return 1;
  }

  file.seekg(0, std::ios::end); 
  std::streamsize size = file.tellg(); 
  file.seekg(0, std::ios::beg); 

  if (frame_size != size){
    std::cerr << "输入文件大小与采样buffer大小不一致" << std::endl;
    return 1;
  }

  std::size_t numShorts = size / sizeof(short);
  std::vector<short> temp_buffer(numShorts);

  if (file.read(reinterpret_cast<char*>(temp_buffer.data()), size)) {
    std::cout << "成功读取文件，大小: " << size << " 字节" << std::endl;
  } else {
    std::cerr << "读取文件失败" << std::endl;
    return 1;
  }
  file.close();

  auto soundClassifcaiton = SoundClassificationV2(dev_id, bmodel_path, threshold);
  auto ret = soundClassifcaiton.onModelOpened();
  int res = -1;
  ret = soundClassifcaiton.inference(temp_buffer.data(), numShorts, &res);
  std::cout << classname[res] << std::endl;

  return 0;
}


