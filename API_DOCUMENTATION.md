# Face-Motion-GPU: Multimodal Sale AI Assessment API

Tài liệu đặc tả kỹ thuật REST API đánh giá kỹ năng bán hàng đa phương thức (Vision + Audio + Fusion + Scoring) trên GPU.

---

## 1. Tổng Quan Kiến Trúc

* **Mục tiêu**: Đánh giá tự động năng lực bán hàng của học viên/nhân viên qua video roleplay (Giao tiếp mắt, Nụ cười, Tư thế đầu, Lắng nghe tích cực, Ngữ điệu giọng nói, Nhận diện tiếng Việt ASR).
* **Cơ chế Warm VRAM**: Nạp sẵn 5 mô hình AI vào GPU VRAM khi khởi động server, **triệt tiêu 4.5s cold-start**, tốc độ đạt **4.0x - 7.0x Real-time**.
* **Base URL**: `http://<SERVER_IP>:8000`
* **Swagger UI (Interactive Docs)**: `http://<SERVER_IP>:8000/docs`

---

## 2. Danh Sách Endpoints

| Phương thức | Endpoint | Định dạng Input | Mục đích |
| :---: | :--- | :---: | :--- |
| `GET` | `/api/v1/health` | Không | Kiểm tra trạng thái GPU VRAM & Warm state |
| `POST` | `/api/v1/analyze` | `multipart/form-data` | Upload trực tiếp file video (Web/Mobile App) |
| `POST` | `/api/v1/analyze-json` | `application/json` | Truyền đường dẫn file trên server/shared storage |
| `GET` | `/static/results/{file}` | URL Path | Xem/Tải video annotated & báo cáo Markdown |

---

## 3. Chi Tiết Các Endpoints

### 3.1. Health Check & GPU Status
Kiểm tra server đã sẵn sàng nhận request và dung lượng VRAM đang dùng.

* **Request**: `GET /api/v1/health`
* **Response (200 OK)**:
```json
{
  "status": "healthy",
  "service_warm": true,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "vram_used_mb": 1420.5,
  "vram_total_mb": 12288.0
}
```

---

### 3.2. Phân Tích Video Qua Đường Dẫn File (`POST /api/v1/analyze-json`)
*Thích hợp nhất cho Microservices nội bộ cùng chia sẻ ổ đĩa hoặc NFS/S3 mount.*

* **Headers**: `Content-Type: application/json`
* **Request Body**:
```json
{
  "video_path": "/home/dinhnhat/Face-motion-gpu/video/tiktok3.mp4",
  "audio_path": null,
  "step": 6,
  "batch_size": 32,
  "save_video": false,
  "show_hud": true,
  "session_id": "tiktok3"
}
```

#### Bảng tham số:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| `video_path` | `string` | *Bắt buộc* | Đường dẫn tuyệt đối tới file video trên server |
| `step` | `int` | `2` | Bước nhảy frame. `step=6` nhanh gấp ~3 lần `step=2` mà vẫn chuẩn xác |
| `batch_size` | `int` | `32` | Số khuôn mặt đưa vào GPU cùng lúc (Khuyên dùng: `32` cho GPU 8-12GB) |
| `save_video` | `bool` | `false` | `true`: render video có thanh HUD cảm xúc; `false`: chỉ lấy dữ liệu JSON (nhanh nhất) |
| `show_hud` | `bool` | `true` | Vẽ overlay bảng chỉ số HUD lên video (nếu `save_video=true`) |
| `session_id` | `string` | `null` | Tên phiên đánh giá (mặc định lấy theo tên file video) |

---

### 3.3. Phân Tích Video Qua Upload Trực Tiếp (`POST /api/v1/analyze`)
*Thích hợp cho Web Frontend, Mobile App hoặc Postman gửi thẳng video lên.*

* **Headers**: `Content-Type: multipart/form-data`
* **Form Fields**:
  * `file`: File video đính kèm (`.mp4`, `.mov`, `.avi`...).
  * `step`: `6` (tùy chọn).
  * `batch_size`: `32` (tùy chọn).
  * `save_video`: `false` (tùy chọn).

---

## 4. Cấu Trúc Dữ Liệu Phản Hồi (Response JSON)

```json
{
  "session_id": "tiktok3",
  "metadata": {
    "session_id": "tiktok3",
    "duration_seconds": 38.6,
    "total_frames_analyzed": 193,
    "total_turns": 5,
    "speaking_turns": 3,
    "listening_turns": 2,
    "total_words_spoken": 140,
    "annotated_video_path": null
  },
  "overall_evaluation": {
    "total_score": 70.5,
    "max_score": 100.0,
    "grade": "A (Thành Thạo)",
    "breakdown": {
      "confidence_score": 21.0,
      "active_listening_score": 22.0,
      "vocal_dynamism_score": 18.0,
      "facial_warmth_score": 9.5
    },
    "strengths": [
      "Tự tin, mạch lạc, hầu như không có khoảng ngập ngừng lúng túng.",
      "Gật đầu đồng thuận rất tốt (21 lần), thể hiện sự lắng nghe và thấu hiểu."
    ],
    "areas_for_improvement": [
      "Tỷ lệ nhìn lơ đễnh đi chỗ khác khá cao (39%).",
      "Gương mặt còn nghiêm nghị, nên mỉm cười chào và cảm ơn khách hàng."
    ],
    "criteria_details": {
      "confidence": { "score": 21.0, "max": 25, "feedback": ["Cần tăng thời lượng nhìn thẳng."] },
      "active_listening": { "score": 22.0, "max": 25, "feedback": [] },
      "vocal_dynamism": { "score": 18.0, "max": 25, "feedback": ["Cần thêm nhiệt huyết vào chất giọng."] },
      "facial_warmth": { "score": 9.5, "max": 25, "feedback": ["Cần mỉm cười tươi tắn hơn."] }
    }
  },
  "final_behavioral_metrics": {
    "facial_emotions": {
      "happy_ratio": 0.12,
      "neutral_ratio": 0.65,
      "sad_ratio": 0.10,
      "negative_ratio": 0.23,
      "average_valence": 0.18
    },
    "attention_and_focus": {
      "focus_ratio": 0.85,
      "distracted_ratio": 0.15,
      "longest_focus_streak_seconds": 12.4,
      "longest_distraction_seconds": 2.8,
      "longest_distraction_episode": {
        "start_time": 15.2,
        "end_time": 18.0,
        "duration": 2.8,
        "reason": "Cúi đầu nhìn xuống (Pitch: -18.2°); Mắt hướng về 'down'"
      },
      "eye_contact_ratio": 0.78,
      "nodding_count": 21
    },
    "vocal_interaction": {
      "speech_ratio": 0.75,
      "pause_ratio": 0.25,
      "speech_rate_wps": 3.6,
      "pitch_variance": 42.5,
      "vocal_enthusiasm_ratio": 0.18
    }
  },
  "key_behavioral_events": {
    "longest_distraction": {
      "duration_seconds": 2.8,
      "start_time": 15.2,
      "end_time": 18.0,
      "reason": "Cúi đầu nhìn xuống (Pitch: -18.2°); Mắt hướng về 'down'"
    },
    "distraction_episodes": [ ... ],
    "hesitation_events": [ ... ],
    "nodding_moments": [ ... ]
  },
  "conversational_turns": [
    {
      "turn_id": 1,
      "type": "speaking",
      "start_time": 0.0,
      "end_time": 30.15,
      "duration": 30.15,
      "text": "MỘT CON TÔM HÙM BÔNG BÊN ANH...",
      "vision": { "focus_ratio": 0.67, "eye_contact_ratio": 0.69, "dominant_emotion": "neutral" },
      "audio": { "is_speaking": true, "word_count": 131 }
    }
  ],
  "timeline_1s": [
    {
      "second": 0,
      "time_range": [0.0, 1.0],
      "vision": { "is_focused": true, "dominant_emotion": "neutral", "valence": 0.1, "head_yaw": 2.1, "head_pitch": -1.0, "gaze": "camera" },
      "audio": { "is_speech": true, "pitch": 210.5, "energy": 0.15, "tone": "neutral" }
    }
  ],
  "files": {
    "json_report": "results/tiktok3_report.json",
    "markdown_report": "results/tiktok3_report.md",
    "annotated_video_url": "/static/results/tiktok3_annotated.mp4"
  },
  "performance": {
    "execution_time_sec": 6.82,
    "realtime_multiplier": "5.66x"
  }
}
```

---

## 5. Ví Dụ Tích Hợp (Code Snippets)

### 5.1. Gọi bằng cURL (Terminal)
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-json" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/home/dinhnhat/Face-motion-gpu/video/tiktok3.mp4",
    "step": 6,
    "batch_size": 32,
    "save_video": false
  }'
```

### 5.2. Gọi bằng Python (Microservice / Backend)
```python
import requests

url = "http://localhost:8000/api/v1/analyze-json"
payload = {
    "video_path": "/home/dinhnhat/Face-motion-gpu/video/tiktok3.mp4",
    "step": 6,
    "batch_size": 32,
    "save_video": False
}

res = requests.post(url, json=payload).json()
print(f"Điểm số: {res['overall_evaluation']['total_score']}/100 ({res['overall_evaluation']['grade']})")
print(f"Thời gian chạy: {res['performance']['execution_time_sec']}s ({res['performance']['realtime_multiplier']})")
```

### 5.3. Gọi bằng Node.js / JavaScript (Frontend / Backend)
```javascript
const axios = require('axios');

async function evaluateVideo() {
  const res = await axios.post('http://localhost:8000/api/v1/analyze-json', {
    video_path: '/home/dinhnhat/Face-motion-gpu/video/tiktok3.mp4',
    step: 6,
    batch_size: 32,
    save_video: false
  });

  const report = res.data;
  console.log(`Điểm: ${report.overall_evaluation.total_score} - Hạng: ${report.overall_evaluation.grade}`);
  console.log(`Lượt nói: ${report.metadata.total_turns} lượt`);
}
evaluateVideo();
```

---

## 6. Mã Lỗi Thường Gặp (Error Codes)

| Mã lỗi | Nguyên nhân | Hướng xử lý |
| :---: | :--- | :--- |
| `400 Bad Request` | Thiếu cả `file` và `video_path` | Cung cấp ít nhất 1 nguồn video |
| `404 Not Found` | Đường dẫn `video_path` không tồn tại trên server | Kiểm tra lại đường dẫn file trên ổ đĩa |
| `503 Service Unavailable` | Các model đang trong quá trình nạp vào GPU | Đợi server warm-up xong (~4s) rồi gọi lại |
| `500 Internal Error` | Lỗi trong quá trình giải mã video hoặc inference | Kiểm tra định dạng codec video hoặc log server |
