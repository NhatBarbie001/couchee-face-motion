# Face-Motion-GPU: Multimodal Sale AI Assessment API

Tài liệu đặc tả kỹ thuật REST API đánh giá kỹ năng bán hàng & giao tiếp đa phương thức (Vision + Audio + Fusion + Prosody) trên GPU.

---

## 1. Tổng Quan Kiến Trúc

* **Mục tiêu**: Đóng vai trò là **Sensory Perception Engine** (Bộ cảm biến giác quan khách quan) phân tích video/audio buổi học Roleplay, bóc tách các bằng chứng hành vi (Giao tiếp mắt, Nụ cười, Cúi/ngẩng đầu, Lắng nghe tích cực, Ngữ điệu, Âm lượng, Tốc độ nói WPM) phục vụ cho LLM-as-a-Judge hoặc Dashboard chấm điểm.
* **Cơ chế Warm VRAM**: Nạp sẵn các mô hình AI vào GPU VRAM khi khởi động server, **triệt tiêu 4.5s cold-start**, tốc độ đạt **6.0x – 19.0x Real-time**.
* **Hỗ trợ 2 chế độ**:
  1. `media_type == "video"`: Phân tích toàn diện cả Hình ảnh (Webcam) + Giọng nói (Microphone).
  2. `media_type == "audio"`: Tự động bỏ qua Vision GPU khi học viên tắt camera, chỉ phân tích âm thanh siêu tốc (~19x Real-time).
* **Base URL**: `http://<SERVER_IP>:8000`
* **Swagger UI (Interactive Docs)**: `http://<SERVER_IP>:8000/docs`

---

## 2. Danh Sách Endpoints

| Phiên bản | Phương thức | Endpoint | Định dạng Input | Mục đích |
| :---: | :---: | :--- | :---: | :--- |
| **v2 (Auto Turns)** | `POST` | `/api/v2/analyze-auto` | `application/json` | **Sensory Auto-Turns**: Tự động bóc tách lượt nói bằng ASR + VAD 0.3s (không cần `turn_markers`) |
| **v2 (Auto Turns)** | `POST` | `/api/v2/analyze-video` | `multipart/form-data` | Upload video trực tiếp hoặc truyền path, tự động ngắt lượt khi nghỉ $\ge 0.3$s |
| **v2 (Khuyên dùng)** | `POST` | `/api/v2/analyze` | `application/json` | **Sensory Engine v2**: Bóc tách bằng chứng hành vi theo `turn_markers` có sẵn |
| **System** | `GET` | `/api/v1/health` | Không | Kiểm tra trạng thái GPU VRAM & Warm state |
| **v1 (Legacy)** | `POST` | `/api/v1/analyze-json` | `application/json` | Phân tích video đường dẫn file (Chấm điểm cứng 100đ) |
| **v1 (Legacy)** | `POST` | `/api/v1/analyze` | `multipart/form-data` | Upload trực tiếp file video từ Form |
| **Static** | `GET` | `/static/results/{file}` | URL Path | Tải video annotated & báo cáo |

---

## 3. Chi Tiết API v2: Sensory Engine (`POST /api/v2/analyze`)

Endpoint này được thiết kế dành riêng cho **Hệ thống Roleplay Coach Couchee**, tiếp nhận đường dẫn file và danh sách các mốc thời gian hội thoại (`turn_markers`), trả về bằng chứng giác quan khách quan mà **không gán cứng điểm số**.

### 3.1. Request Body (`AnalyzeRequestV2`)

* **Headers**: `Content-Type: application/json`
* **Cấu trúc JSON**:
```json
{
  "session_id": "roleplay_sales_38s",
  "video_path": "/home/dinhnhat/couchee-face-motion/video/tiktok3.mp4",
  "media_type": "video",
  "step": 6,
  "batch_size": 32,
  "include_timeline_1s": false,
  "turn_markers": [
    {
      "role": "student",
      "start_sec": 0.0,
      "end_sec": 6.5,
      "text": "Dạ em chào anh ạ, em là Tuấn chuyên viên tư vấn bên Couchee!"
    },
    {
      "role": "ai",
      "start_sec": 6.5,
      "end_sec": 13.0,
      "text": "Chào em, bên anh đang muốn tìm giải pháp đào tạo bán hàng bằng AI."
    }
  ]
}
```

#### Bảng tham số:
| Tham số | Kiểu | Mặc định | Bắt buộc | Ý nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| `video_path` | `string` | — | **Có** | Đường dẫn tuyệt đối tới file `.mp4`, `.webm`, `.wav` trên server |
| `session_id` | `string` | `null` | Không | Mã phiên học (mặc định lấy theo tên file) |
| `media_type` | `string` | `"video"` | Không | `"video"` (chạy cả cam & mic) hoặc `"audio"` (chế độ tắt cam) |
| `turn_markers` | `list` | `[]` | Không | Danh sách các lượt đối đáp giữa học viên và AI |
| `turn_markers[].role` | `string` | — | **Có** | `'student'` (học viên nói) hoặc `'ai'` (AI khách hàng nói) |
| `turn_markers[].start_sec` | `float` | — | **Có** | Thời điểm bắt đầu câu nói (tính bằng giây) |
| `turn_markers[].end_sec` | `float` | — | **Có** | Thời điểm kết thúc câu nói (tính bằng giây) |
| `turn_markers[].text` | `string` | `null` | Không | Lời thoại câu nói nếu có |
| `step` | `int` | `6` | Không | Bước nhảy frame video. `step=6` nhanh gấp ~3 lần `step=2` mà vẫn chính xác |
| `batch_size` | `int` | `32` | Không | Kích thước batch đưa vào GPU (khuyên dùng: 32) |
| `include_timeline_1s` | `bool` | `false` | Không | `true`: trả về mảng dữ liệu chi tiết từng giây; `false`: chỉ lấy tổng quan |

---

### 3.2. Response Body (`BehavioralEvidenceResponse`)

```json
{
  "session_id": "roleplay_sales_38s",
  "status": "completed",
  "media_type": "video",
  "duration_seconds": 38.6,
  "performance": {
    "execution_time_sec": 6.64,
    "realtime_multiplier": "5.81x"
  },
  "overall_metrics": {
    "eye_contact_ratio": 0.6114,
    "distracted_ratio": 0.4145,
    "nodding_count": 12,
    "smile_ratio": 0.0,
    "dominant_emotions": {
      "happy": 0.0,
      "neutral": 0.1865,
      "sad": 0.3264,
      "negative": 0.715,
      "average_valence": -0.633
    },
    "speech_rate_wpm": 217.7,
    "pitch_variance": 9000.16,
    "speech_ratio": 0.8367,
    "pause_ratio": 0.1633,
    "total_hesitations_count": 1,
    "energy_mean": 0.0482,
    "vocal_enthusiasm_ratio": 0.1845,
    "vocal_tone": "confident"
  },
  "anomalies": {
    "distraction_moments": [
      {
        "start_sec": 28.8,
        "end_sec": 31.6,
        "duration": 2.8,
        "reason": "Quay đầu sang phải (Yaw: 19.2°); Mắt hướng về 'right'"
      }
    ],
    "hesitation_moments": [
      {
        "start_sec": 32.91,
        "end_sec": 38.0,
        "duration": 5.09,
        "reason": "Ngập ngừng ngắt quãng kéo dài 5.09s"
      }
    ],
    "nodding_moments": [
      {
        "timestamp": 7.0,
        "amplitude": 11.4
      }
    ]
  },
  "turn_evidence": [
    {
      "turn_index": 1,
      "role": "student",
      "start_sec": 0.0,
      "end_sec": 6.5,
      "duration_sec": 6.5,
      "text": "Dạ em chào anh ạ, em là Tuấn chuyên viên tư vấn bên Couchee!",
      "eye_contact_ratio": 0.8182,
      "smile_ratio": 0.0,
      "dominant_emotion": "neutral",
      "nodding_count": null,
      "attentive_gaze_ratio": null,
      "speech_rate_wpm": 212.3,
      "hesitation_seconds": 0.0,
      "is_speaking": true,
      "energy_mean": 0.0512,
      "vocal_tone": "confident"
    },
    {
      "turn_index": 2,
      "role": "ai",
      "start_sec": 6.5,
      "end_sec": 13.0,
      "duration_sec": 6.5,
      "text": "Chào em, bên anh đang muốn tìm giải pháp đào tạo bán hàng bằng AI.",
      "eye_contact_ratio": 0.7241,
      "smile_ratio": 0.0,
      "dominant_emotion": "neutral",
      "nodding_count": 3,
      "attentive_gaze_ratio": 0.7576,
      "speech_rate_wpm": null,
      "hesitation_seconds": 0.0,
      "is_speaking": false,
      "energy_mean": null,
      "vocal_tone": null
    }
  ],
  "timeline_1s": null
}
```

---

### 3.3. Giải Thích Ý Nghĩa Các Chỉ Số Giác Quan

#### 1. Nhóm chỉ số toàn phiên (`overall_metrics`)
* **`eye_contact_ratio`**: Tỷ lệ duy trì ánh mắt nhìn thẳng vào camera ($0.0 \rightarrow 1.0$). Chuẩn bán hàng: $\ge 0.70$ (70%).
* **`distracted_ratio`**: Tỷ lệ thời gian quay đầu hoặc nhìn lơ đãng ra ngoài. Chuẩn: $\le 0.20$ (20%).
* **`nodding_count`**: Tổng số lần gật đầu thể hiện sự tiếp thu/đồng thuận (đã lọc với ngưỡng biên độ $\ge 11.0^\circ$).
* **`smile_ratio`**: Tỷ lệ thời gian mỉm cười thân thiện ($0.0 \rightarrow 1.0$).
* **`energy_mean`**: Năng lượng âm lượng trung bình RMS. Đo mức độ **nói to, rõ ràng, nội lực** hay **nói lí nhí, thì thầm**.
* **`vocal_enthusiasm_ratio`**: Tỷ lệ giọng nói hào hứng, nhiệt huyết.
* **`vocal_tone`**: Phân loại chất giọng thực tế thành **4 nhóm chuẩn**:
  * `'enthusiastic'`: Giọng hào hứng, nhiệt huyết, biến thiên cao độ sinh động.
  * `'confident'`: Giọng đĩnh đạc, tự tin, trường âm và tốc độ cân bằng chuẩn mực.
  * `'monotone'`: Giọng đều đều buồn ngủ, thiếu nhấn nhá (pitch variance $< 2500$).
  * `'nervous'`: Giọng run rẩy, cao độ bất thường hoặc ngập ngừng nhiều.
* **`speech_rate_wpm`**: Tốc độ nói tính bằng Từ/Phút (Words Per Minute). Chuẩn lý tưởng: 140 – 180 WPM.
* **`speech_ratio` / `pause_ratio`**: Tỷ lệ thời lượng nói so với thời lượng im lặng/lắng nghe.

#### 2. Nhóm bằng chứng theo lượt thoại (`turn_evidence`)
* **Lượt Học viên (`student`)**: Đo `eye_contact_ratio`, `smile_ratio`, `speech_rate_wpm`, `energy_mean`, `vocal_tone`, `hesitation_seconds`.
* **Lượt AI (`ai`)**: Đo thái độ lắng nghe của học viên: `attentive_gaze_ratio` (mắt chú ý), `nodding_count` (số lần gật đầu), `smile_ratio` (nụ cười khi lắng nghe).

#### 3. Nhóm khoảnh khắc bất thường (`anomalies`)
* **`distraction_moments`**: Mốc giây chính xác học viên quay đầu hoặc liếc mắt đi chỗ khác kèm góc độ và lý do (dùng để gắn cờ trên video seekbar).
* **`hesitation_moments`**: Mốc giây học viên bị ngắc ngứ, ngập ngừng kéo dài $>1.5s$.
* **`nodding_moments`**: Danh sách chi tiết từng cú gật đầu kèm thời điểm và biên độ góc.

---

## 4. API v1 (Legacy): Chấm Điểm Cứng 100đ

### 4.1. Endpoint `POST /api/v1/analyze-json`
Nhận đường dẫn file video và tự động tính điểm 100đ theo rubric truyền thống.

* **Request Body**:
```json
{
  "video_path": "/home/dinhnhat/couchee-face-motion/video/tiktok3.mp4",
  "step": 6,
  "batch_size": 32,
  "save_video": false,
  "show_hud": true
}
```

* **Response Trả về**:
  * `total_score`: Điểm tổng kết (thang 100).
  * `grade`: Xếp loại (`A (Thành Thạo)`, `B`, `C`...).
  * `breakdown`: Điểm 4 tiêu chí (Độ tự tin 25đ, Lắng nghe tích cực 25đ, Ngữ điệu giọng nói 25đ, Sự ấm áp nét mặt 25đ).
  * `strengths` & `areas_for_improvement`: Nhận xét mẫu.

---

## 4. Chi Tiết API v2 Auto Turns: Tự Động Trích Xuất Lượt Nói (`POST /api/v2/analyze-auto` & `/api/v2/analyze-video`)

Được thiết kế cho các kịch bản **không có sẵn mốc `turn_markers`** (ví dụ: upload video trực tiếp từ client). Hệ thống sử dụng **Zipformer ASR** nhận diện lời thoại tiếng Việt và **Silero VAD độ nhạy cao (threshold = 0.3)**. Mỗi khi người nói dừng/nghỉ $\ge 0.3$ giây, hệ thống lập tức chốt xong lượt nói hiện tại và tạo lượt nói tiếp theo. Tất cả các turn đều mang `role: "speaker"` (không chèn role AI giả lập).

### 4.1. Endpoint 1: JSON Body (`POST /api/v2/analyze-auto`)
* **Headers**: `Content-Type: application/json`
* **Request Body (`AnalyzeAutoRequest`)**:
```json
{
  "video_path": "/home/dinhnhat/couchee-face-motion/video/tiktok1.mp4",
  "session_id": "auto_session_01",
  "vad_threshold": 0.3,
  "pause_threshold_sec": 0.3,
  "step": 6,
  "batch_size": 32,
  "include_timeline_1s": true
}
```

| Tham số | Kiểu | Mặc định | Bắt buộc | Mô tả |
| :--- | :---: | :---: | :---: | :--- |
| `video_path` | `string` | — | **Có** | Đường dẫn tuyệt đối hoặc tương đối tới video/audio trên server |
| `session_id` | `string` | `null` | Không | Mã định danh phiên làm việc (mặc định lấy theo tên file) |
| `vad_threshold` | `float` | `0.3` | Không | Ngưỡng kích hoạt tiếng nói của Silero VAD (0.3: cực nhạy, bắt âm thì thầm) |
| `pause_threshold_sec`| `float` | `0.3` | Không | Thời gian khoảng lặng (giây) để tách lượt nói riêng biệt (mặc định 0.3s) |
| `step` | `int` | `6` | Không | Bước nhảy frame video |
| `batch_size` | `int` | `32` | Không | Batch size GPU inference |
| `include_timeline_1s` | `bool` | `true` | Không | Trả về chuỗi dữ liệu 1s resolution |

---

### 4.2. Endpoint 2: Multipart Upload (`POST /api/v2/analyze-video`)
* **Headers**: `Content-Type: multipart/form-data`
* **Form Fields**:
  * `file`: File video upload trực tiếp (tuỳ chọn nếu đã có `video_path`).
  * `video_path`: Đường dẫn file video trên server (tuỳ chọn nếu upload `file`).
  * `session_id`: Mã phiên làm việc.
  * `vad_threshold`: Ngưỡng nhạy VAD (mặc định `0.3`).
  * `pause_threshold_sec`: Thời gian nghỉ để tách turn (mặc định `0.3`).
  * `step`: Bước nhảy frame (mặc định `6`).
  * `batch_size`: Batch size GPU (mặc định `32`).
  * `include_timeline_1s`: Kèm timeline 1s (mặc định `true`).

---

### 4.3. Response Body (`BehavioralEvidenceResponse`)
Cả 2 endpoint trả về cùng cấu trúc JSON chuẩn của `/api/v2/analyze`, trong đó `turn_evidence` chứa toàn bộ các lượt nói đã được bóc tách:
```json
{
  "session_id": "auto_session_01",
  "status": "completed",
  "media_type": "video",
  "duration_seconds": 38.6,
  "performance": {
    "execution_time_sec": 6.2,
    "realtime_multiplier": "6.22x"
  },
  "overall_metrics": {
    "eye_contact_ratio": 0.825,
    "distracted_ratio": 0.175,
    "nodding_count": 5,
    "smile_ratio": 0.35,
    "dominant_emotions": { "happy": 0.35, "neutral": 0.65 },
    "speech_rate_wpm": 165.2,
    "pitch_variance": 4820.5,
    "vocal_tone": "confident"
  },
  "anomalies": {
    "distraction_moments": [],
    "hesitation_moments": [],
    "nodding_moments": []
  },
  "turn_evidence": [
    {
      "turn_index": 1,
      "role": "speaker",
      "start_sec": 0.42,
      "end_sec": 4.18,
      "duration_sec": 3.76,
      "text": "Chào mừng quý khách đã đến với cửa hàng của chúng tôi",
      "eye_contact_ratio": 0.88,
      "smile_ratio": 0.45,
      "dominant_emotion": "happy",
      "nodding_count": 1,
      "attentive_gaze_ratio": 0.92,
      "speech_rate_wpm": 175.5,
      "hesitation_seconds": 0.0,
      "is_speaking": true,
      "energy_mean": 0.045,
      "vocal_tone": "confident"
    }
  ],
  "timeline_1s": [ ... ]
}
```

---

## 5. Ví Dụ Tích Hợp (Code Snippets)

### 5.1. Test Bằng Python (Requests)

```python
import requests

url = "http://localhost:8000/api/v2/analyze"
payload = {
    "session_id": "session_001",
    "video_path": "/path/to/video.mp4",
    "media_type": "video", # hoặc "audio" nếu tắt cam
    "turn_markers": [
        {"role": "student", "start_sec": 0.0, "end_sec": 10.0, "text": "Lời chào"},
        {"role": "ai", "start_sec": 10.0, "end_sec": 20.0, "text": "Phản hồi"}
    ]
}

res = requests.post(url, json=payload).json()
print("Tông giọng:", res["overall_metrics"]["vocal_tone"])
print("Tốc độ nói:", res["overall_metrics"]["speech_rate_wpm"], "WPM")
print("Giao tiếp mắt:", res["overall_metrics"]["eye_contact_ratio"] * 100, "%")
```

### 5.2. Test Bằng cURL

```bash
curl -X POST "http://localhost:8000/api/v2/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/home/dinhnhat/couchee-face-motion/video/tiktok3.mp4",
    "media_type": "video",
    "turn_markers": [
      {"role": "student", "start_sec": 0.0, "end_sec": 15.0},
      {"role": "ai", "start_sec": 15.0, "end_sec": 30.0}
    ]
  }'
```

### 5.3. Tích Hợp Từ Couchee Backend (Moleculer Action)

```javascript
const axios = require('axios');

module.exports = {
  name: 'roleplay.facemotion',
  actions: {
    async extractBehavioralEvidence(ctx) {
      const { videoPath, mediaType, turnMarkers, sessionId } = ctx.params;
      const gpuUrl = process.env.FACE_MOTION_GPU_URL || 'http://localhost:8000';

      try {
        const response = await axios.post(`${gpuUrl}/api/v2/analyze`, {
          session_id: sessionId,
          video_path: videoPath,
          media_type: mediaType || 'video',
          turn_markers: turnMarkers || [],
          include_timeline_1s: false
        }, { timeout: 60000 });

        return response.data;
      } catch (err) {
        this.logger.warn('[Face-motion-gpu] Offline or error:', err.message);
        return null; // Fallback an toàn, không làm gián đoạn bài chấm của LLM
      }
    }
  }
};
```

---

## 6. Mã Lỗi Thường Gặp (Error Codes)

| Mã lỗi | Nguyên nhân | Hướng xử lý |
| :---: | :--- | :--- |
| `404 Not Found` | Đường dẫn `video_path` không tồn tại trên server Linux | Kiểm tra lại đường dẫn file trên ổ đĩa |
| `503 Service Unavailable` | Các model đang trong quá trình warm-up nạp vào GPU | Đợi server warm-up xong (~4 giây) rồi gọi lại |
| `500 Internal Error` | File media hỏng hoặc codec không hỗ trợ | Kiểm tra định dạng file (khuyên dùng `.mp4`, `.webm`, `.wav`) |
| `Connection Refused` | Server `server.py` chưa được bật trên server | Kích hoạt virtualenv và chạy `python server.py` |
