import os
import json
import time
import requests

# ==============================================================================
# CẤU HÌNH THÔNG SỐ TEST
# ==============================================================================
BASE_URL = "http://localhost:8000"  # Hoặc địa chỉ IP/domain server GPU của bạn

# Đường dẫn video trên server (cho API POST /api/v2/analyze-auto)
SERVER_VIDEO_PATH = "/home/dinhnhat/couchee-face-motion/video/sample_video.mp4"

# Đường dẫn file video trên máy hiện tại để upload (cho API POST /api/v2/analyze-video)
LOCAL_VIDEO_PATH = "sample_video.mp4"


# ==============================================================================
# 1. TEST POST /api/v2/analyze-auto (Gửi JSON Path)
# ==============================================================================
def test_analyze_auto_json():
    print("=" * 70)
    print(">>> 1. TEST POST /api/v2/analyze-auto (JSON Body)")
    print("=" * 70)
    
    url = f"{BASE_URL}/api/v2/analyze-auto"
    payload = {
        "video_path": SERVER_VIDEO_PATH,
        "session_id": "test_auto_json_session",
        "vad_threshold": 0.3,          # Độ nhạy VAD (0.3: bắt âm thì thầm)
        "pause_threshold_sec": 0.3,    # Khoảng dừng >= 0.3s sẽ ngắt tạo turn mới
        "step": 6,                     # Frame step
        "batch_size": 32,
        "include_timeline_1s": True
    }
    
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        elapsed = time.time() - start_time
        print(f"Status Code: {response.status_code} (Thời gian gọi: {elapsed:.2f}s)\n")

        try:
            res_json = response.json()
            # In ra toàn bộ JSON trả về định dạng đẹp, giữ nguyên tiếng Việt
            print(json.dumps(res_json, indent=2, ensure_ascii=False))
        except Exception:
            print(response.text)

    except Exception as e:
        print(f"Lỗi kết nối / ngoại lệ: {e}")


# ==============================================================================
# 2. TEST POST /api/v2/analyze-video (Multipart File Upload)
# ==============================================================================
def test_analyze_video_upload():
    print("\n" + "=" * 70)
    print(">>> 2. TEST POST /api/v2/analyze-video (Multipart Upload)")
    print("=" * 70)
    
    url = f"{BASE_URL}/api/v2/analyze-video"
    
    if not os.path.exists(LOCAL_VIDEO_PATH):
        print(f"[!] Không tìm thấy file local: '{LOCAL_VIDEO_PATH}'")
        print("    Vui lòng cập nhật biến LOCAL_VIDEO_PATH để test upload.")
        return

    form_data = {
        "session_id": "test_upload_session",
        "vad_threshold": 0.3,
        "pause_threshold_sec": 0.3,
        "step": 6,
        "batch_size": 32,
        "include_timeline_1s": "true"
    }

    start_time = time.time()
    try:
        with open(LOCAL_VIDEO_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(LOCAL_VIDEO_PATH), f, "video/mp4")
            }
            print(f"Đang upload và xử lý file: {LOCAL_VIDEO_PATH} ...")
            response = requests.post(url, data=form_data, files=files, timeout=300)

        elapsed = time.time() - start_time
        print(f"Status Code: {response.status_code} (Thời gian gọi: {elapsed:.2f}s)\n")

        try:
            res_json = response.json()
            # In ra toàn bộ JSON trả về định dạng đẹp, giữ nguyên tiếng Việt
            print(json.dumps(res_json, indent=2, ensure_ascii=False))
        except Exception:
            print(response.text)

    except Exception as e:
        print(f"Lỗi kết nối / ngoại lệ: {e}")


if __name__ == "__main__":
    # Test 1: Gửi JSON path video trên server
    test_analyze_auto_json()

    # Test 2: Upload trực tiếp file video (bỏ comment dòng dưới nếu muốn test)
    # test_analyze_video_upload()
