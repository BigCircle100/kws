#include "melspec.hpp"
#include <iostream>

using namespace melspec;

static Matrixf melfilter(int sr, int n_fft, int n_mels, int fmin, int fmax, bool htk) {
  int n_f = n_fft / 2 + 1;
  Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f - 1)) * sr) / n_fft;

  float f_min = 0.f;
  float f_sp = 200.f / 3.f;
  float min_log_hz = 1000.f;
  float min_log_mel = (min_log_hz - f_min) / f_sp;
  float logstep = logf(6.4f) / 27.f;

  auto hz_to_mel = [=](int hz, bool htk) -> float {
    if (htk) {
      return 2595.0f * log10f(1.0f + hz / 700.0f);
    }
    float mel = (hz - f_min) / f_sp;
    if (hz >= min_log_hz) {
      mel = min_log_mel + logf(hz / min_log_hz) / logstep;
    }
    return mel;
  };
  auto mel_to_hz = [=](Vectorf &mels, bool htk) -> Vectorf {
    if (htk) {
      return 700.0f *
             (Vectorf::Constant(n_mels + 2, 10.f).array().pow(mels.array() / 2595.0f) - 1.0f);
    }
    return (mels.array() > min_log_mel)
        .select(((mels.array() - min_log_mel) * logstep).exp() * min_log_hz,
                (mels * f_sp).array() + f_min);
  };

  float min_mel = hz_to_mel(fmin, htk);
  float max_mel = hz_to_mel(fmax, htk);
  Vectorf mels = Vectorf::LinSpaced(n_mels + 2, min_mel, max_mel);
  Vectorf mel_f = mel_to_hz(mels, htk);
  Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
  Matrixf ramps =
      mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();

  Matrixf lower = -ramps.topRows(n_mels).array() /
                  fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
  Matrixf upper = ramps.bottomRows(n_mels).array() /
                  fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
  Matrixf weights = (lower.array() < upper.array()).select(lower, upper).cwiseMax(0);

  auto enorm = (2.0 / (mel_f.segment(2, n_mels) - mel_f.segment(0, n_mels)).array())
                   .transpose()
                   .replicate(1, n_f);
  weights = weights.array() * enorm;

  return weights.transpose();
}


MelFeatureExtract::MelFeatureExtract(int num_frames, int sr, int n_fft, int n_hop, int n_mel,
                                     int fmin, int fmax, const std::string &mode, bool htk,
                                     bool center /*=true*/, int power /*=2*/,
                                     bool is_log /*=true*/) {
  num_wav_len_ = num_frames;
  num_fft_ = n_fft;
  num_mel_ = n_mel;
  sample_rate_ = sr;
  num_hop_ = n_hop;
  fmin_ = fmin;
  fmax_ = fmax;
  center_ = center;
  power_ = power;
  mode_ = mode;
  int pad_len = center ? n_fft / 2 : 0;
  pad_len_ = pad_len;

  window_ =
      0.5 *
      (1.f - (Vectorf::LinSpaced(n_fft, 0.f, static_cast<float>(n_fft - 1)) * 2.f * M_PI / n_fft)
                 .array()
                 .cos());
  mel_basis_ = melfilter(sr, n_fft, n_mel, fmin, fmax, htk);
  is_log_ = is_log;
}
MelFeatureExtract::~MelFeatureExtract() {
  if (mp_sft_mag_vec_ != nullptr) {
    delete mp_sft_mag_vec_;
    mp_sft_mag_vec_ = nullptr;
  }
}


static Matrixcf stft(Vectorf &x_paded, Vectorf &window, int n_fft, int n_hop,
                     const std::string &win, bool center, const std::string &mode) {
  // hanning
  // Vectorf window = 0.5*(1.f-(Vectorf::LinSpaced(n_fft, 0.f,
  // static_cast<float>(n_fft-1))*2.f*M_PI/n_fft).array().cos());

  // int pad_len = center ? n_fft / 2 : 0;
  // Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);

  int n_f = n_fft / 2 + 1;
  int n_frames = 1 + (x_paded.size() - n_fft) / n_hop;
  Matrixcf X(n_frames, n_fft);
  Eigen::FFT<float> fft;

  for (int i = 0; i < n_frames; ++i) {
    Vectorf segment = x_paded.segment(i * n_hop, n_fft);
    Vectorf x_frame = window.array() * x_paded.segment(i * n_hop, n_fft).array();
    X.row(i) = fft.fwd(x_frame);
  }
  return X.leftCols(n_f);
}

static Matrixf spectrogram(Matrixcf &X, float power = 1.f) {
  return X.cwiseAbs().array().pow(power);
}

void MelFeatureExtract::update_data(short *p_data, int data_len) {
  if (x_pad_.cols() == 0) {
    x_pad_ = Vectorf::Constant(pad_len_ * 2 + data_len, 0);
  }
  int num_len = int(x_pad_.size()) - 2 * pad_len_;
  if (num_len != data_len) {
    std::cerr << "update_data size error" << std::endl;
  }
  for (int i = 0; i < data_len; i++) {
    x_pad_[i + pad_len_] = p_data[i] / 32768.0;
  }
  int left = pad_len_;
  int right = pad_len_;
  if (mode_.compare("reflect") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 2 - i + left] / 32768.0;
    }
  }

  if (mode_.compare("symmetric") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i - 1] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1 - i + left] / 32768.0;
    }
  }

  if (mode_.compare("edge") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[0] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1] / 32768.0;
    }
  }
}

void MelFeatureExtract::melspectrogram_optimze(short *p_data, int data_len, float *p_dst,
                                               int dst_len, float q_scale, bool fix, float eps,
                                               float s, float alpha, float delta, float r) {
  int pad_len = center_ ? num_fft_ / 2 : 0;

  int n_f = num_fft_ / 2 + 1;
  int padded_len = data_len + 2 * pad_len;
  int n_frames = 1 + (padded_len - num_fft_) / num_hop_;

  Eigen::FFT<float> fft;
  melspec::Vectorf segment(num_fft_);
  melspec::Vectorf specmag(n_f);
  melspec::Vectorf last_state(num_mel_);

  const float scale = 1.0 / 32768.0;
  for (int i = 0; i < n_frames; ++i) {
    int start_idx = i * num_hop_;
    for (int j = 0; j < num_fft_; j++) {
      int srcidx = start_idx + j - pad_len;
      if (srcidx < 0) {
        srcidx = -srcidx;
      } else if (srcidx >= data_len) {
        int over = srcidx - data_len;
        srcidx = data_len - over - 2;
      }
      segment[j] = p_data[srcidx] * scale;  // TODO:fuquan.ke this could be optimized
    }
    melspec::Vectorf x_frame = window_.array() * segment.array();
    melspec::Vectorcf spec_ri = fft.fwd(x_frame);
    // compute mag
    melspec::Vectorf specmag = spec_ri.leftCols(n_f).cwiseAbs().array().pow(2);

    float *pdst_r = p_dst + i * num_mel_;
    melspec::Vectorf rowv = specmag * mel_basis_;

    if (fix) {  // use pcen

      if (i == 0) {
        last_state = rowv;
      } else {
        last_state = (1 - s) * last_state + s * rowv;
      }

      melspec::Vectorf pcen_data =
          (rowv.array() / (last_state.array() + eps).pow(alpha) + delta).pow(r) - pow(delta, r);

      for (int n = 0; n < num_mel_; n++) {
        float qval = pcen_data[n] * q_scale;

        // if (qval < -128) {
        //   qval = -128;
        // } else if (qval > 127) {
        //   qval = 127;
        // }
        pdst_r[n] = qval;
      }
    } else {
      for (int n = 0; n < num_mel_; n++) {
        float v = rowv[n];
        if (v < min_val_) v = min_val_;
        if (is_log_) {
          v = 10 * log10f(v);
        }
        float qval = v * q_scale;
        // if (qval < -128) {
        //   // std::cout<<"overflow qval:"<<qval<<std::endl;
        //   qval = -128;
        // } else if (qval > 127) {
        //   // std::cout<<"overflow qval:"<<qval<<std::endl;
        //   qval = 127;
        // }
        pdst_r[n] = qval;
      }
    }
  }
}
