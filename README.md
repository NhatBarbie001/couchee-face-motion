# Face-motion-gpu: High-Throughput Offline Multimodal Sale AI Pipeline

Hệ thống phân tích đa phương thức (Thị giác + Thính giác + Đánh giá năng lực giao tiếp bán hàng) theo chế độ **Offline Batch Processing**, được tối ưu hóa đặc biệt cho máy chủ cấu hình mạnh (**64 CPU threads, 125 GB RAM, NVIDIA RTX 3060 12GB VRAM**).

---

## 1. Kiến Trúc Hệ Thống

```text
                               SALE TRAINEE VIDEO/AUDIO
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ VISION MODULE ]                               [ AUDIO MODULE ]
    (Decord / Multi-thread Decoders)                (FFmpeg 16kHz Mono Stream)
                  │                                               │
    ┌─────────────┼─────────────┐                   ┌─────────────┼─────────────┐
    ▼             ▼             ▼                   ▼             ▼             ▼
  SCRFD      EmotiEffLib     6DRepNet            Silero       Aniemore       Prosody
Detection     (Emotion)     (Head Pose)            VAD        Wav2Vec2     (Pitch/Energy/
  + Bbox      Batch FP16    Pitch/Yaw/Roll      (Speech)      (Emotion)     Rate/Variance)
    │             │             │                   │             │             │
    └─────────────┼─────────────┘                   └─────────────┼─────────────┘
                  ▼                                               ▼
             Gaze-LLE                                       AudioFeatures
         (Eye Contact/Gaze)                         (Speech ratio, WPM, Pitch...)
                  │                                               │
                  ▼                                               │
            VisionFeatures                                        │
    (Gaze away, Nodding, Valence...)                              │
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                                [ FEATURE FUSION ]
                          (Temporal Alignment & Merging)
                                          │
                                          ▼
                                 [ SALE SCORING ]
                        (Rubric: Confidence, Empathy,
                         Active Listening, Pitch Energy)
                                          │
                                          ▼
                                   [ FINAL REPORT ]
                               (JSON + Markdown Summary)
```

---

## 2. Các Model Thành Phần & Tối Ưu Hóa GPU

| Module | Model | Vai Trò | Tối Ưu Hóa Phần Cứng | VRAM Ước Tính |
| :--- | :--- | :--- | :--- | :---: |
| **Face Detect** | **SCRFD** | Bbox khuôn mặt + 5 điểm mốc landmarks | ONNX TensorRT / CUDAProvider | ~1.2 GB |
| **Face Emotion** | **EmotiEffLib** | Phân loại 8 cảm xúc + Valence | PyTorch FP16 / Batch 32-64 | ~1.5 GB |
| **Head Pose** | **6DRepNet** | Yaw (trái/phải), Pitch (gật/ngửa), Roll | ONNX CUDAProvider / Batch 32 | ~1.0 GB |
| **Eye Gaze** | **Gaze-LLE** | Hướng nhìn 3D & Duy trì Eye Contact | ViT-B FP16 CUDA / Geometric | ~2.5 GB |
| **Audio VAD** | **Silero VAD** | Lọc khoảng lặng, ngắt quãng > 2s | ONNX CUDA / Torch Hub | ~0.5 GB |
| **Voice Emotion** | **Aniemore Wav2Vec2** | Cảm xúc giọng nói tiếng Việt/đa ngữ | Transformers FP16 trên GPU | ~2.0 GB |
| **Prosody** | **Praat / Librosa** | Cao độ Pitch, Năng lượng, Tốc độ nói | Đa luồng CPU (64 threads) | ~0.0 GB (CPU) |
| **TỔNG CỘNG** | | | | **~8.7 GB / 12 GB** |

> **An Toàn Bộ Nhớ:** Tổng lượng VRAM chiếm dụng chỉ ~8.7 GB, nằm hoàn toàn trong dung lượng 12GB VRAM của RTX 3060, không lo bị lỗi Out-Of-Memory (OOM).

---

## 3. Cấu Trúc Thư Mục

```text
Face-motion-gpu/
├── vision/                      # Pipeline thị giác độc lập
│   ├── face_detection/          # SCRFD
│   ├── face_emotion/            # EmotiEffLib
│   ├── head_pose/               # 6DRepNet
│   ├── eye_gaze/                # Gaze-LLE
│   └── pipeline.py              # VisionModule.process(video) -> VisionFeatures
│
├── audio/                       # Pipeline âm thanh độc lập
│   ├── vad/                     # Silero VAD
│   ├── voice_emotion/           # Aniemore Wav2Vec2
│   ├── prosody/                 # Pitch, Energy, WPM
│   └── pipeline.py              # AudioModule.process(media) -> AudioFeatures
│
├── fusion/                      # Tầng hợp nhất đặc trưng
│   ├── feature_fusion.py        # FusedMultimodalFeatures
│   └── temporal.py              # Đồng bộ hóa timeline
│
├── scoring/                     # Bộ chấm điểm năng lực bán hàng (4 trụ cột)
│   ├── rubric.py                # Định nghĩa tiêu chuẩn chấm điểm
│   └── scorer.py                # SaleScorer & Markdown/JSON exporter
│
├── benchmark/                   # Đo lường hiệu năng FPS, RTF & VRAM
│   ├── benchmark_vision.py      # Benchmark các model thị giác
│   └── benchmark_audio.py       # Benchmark các model âm thanh
│
├── models/                      # Quản lý weights & downloader
│   └── download_models.py
│
├── run_pipeline.py              # Master CLI Runner
└── requirements-gpu.txt
```

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### Bước 1: Cài đặt Dependencies
```bash
pip install -r requirements-gpu.txt
```

### Bước 2: Tải hoặc Kiểm tra Checkpoints Model
```bash
python models/download_models.py
```

### Bước 3: Chạy Toàn Bộ Quy Trình Đánh Giá (End-to-End)
```bash
# Phân tích file video bán hàng với GPU RTX 3060 (batch size 32):
python run_pipeline.py --video "C:\Users\loidi\Downloads\tiktok1.mp4" --output-dir "results/demo_run" --device cuda --batch-size 32

# Nếu chạy trên CPU:
python run_pipeline.py --video "C:\Users\loidi\Downloads\tiktok1.mp4" --output-dir "results/demo_run" --device cpu
```

### Bước 4: Chạy Benchmark Hiệu Năng
```bash
# Benchmark Vision Models:
python -m benchmark.benchmark_vision --device cuda --batch-size 32

# Benchmark Audio Models:
python -m benchmark.benchmark_audio --device cuda
```

---

## 5. Sử Dụng Độc Lập Qua Code Python

Hai module hoàn toàn tách biệt, có thể gọi độc lập trong hệ thống khác:

```python
from vision.pipeline import VisionModule
from audio.pipeline import AudioModule
from fusion.feature_fusion import FeatureFusion
from scoring.scorer import SaleScorer

# 1. Chạy Vision Module
vision_module = VisionModule(device="cuda")
vision_features = vision_module.process("video.mp4", batch_size=32)

# 2. Chạy Audio Module
audio_module = AudioModule(device="cuda")
audio_features = audio_module.process("video.mp4")

# 3. Hợp nhất đặc trưng
fused = FeatureFusion.fuse("session_01", vision_features, audio_features)

# 4. Chấm điểm & sinh báo cáo
scorer = SaleScorer()
report = scorer.score_session(fused)
scorer.save_reports(report, output_dir="results/")
```
