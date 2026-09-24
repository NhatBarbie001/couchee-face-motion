import requests
import json
import time
import os

# Server GPU URL
SERVER_URL = "http://192.168.1.22:8000/api/v2/analyze-upload"

# Đường dẫn video test trên máy Windows của bạn
TEST_VIDEO_PATH = r"C:\Users\loidi\OneDrive\Desktop\test.mp4"

def test_upload():
    print("=" * 70)
    print("TESTING DIRECT MULTIPART UPLOAD FROM WINDOWS TO GPU SERVER")
    print("=" * 70)

    # Nếu chưa có file test.mp4, tìm file mp4 bất kỳ trong thư mục Download / Videos
    target_video = TEST_VIDEO_PATH
    if not os.path.exists(target_video):
        # Thử tìm file .mp4 trong Downloads
        download_dir = os.path.expanduser(r"~\Downloads")
        mp4_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".mp4")] if os.path.exists(download_dir) else []
        if mp4_files:
            target_video = mp4_files[0]
            print(f"[*] Tìm thấy file video trong Downloads: {target_video}")
        else:
            print(f"[!] Vui lòng đặt một file video .mp4 tại '{TEST_VIDEO_PATH}' để test.")
            return

    # Giả lập turn_markers từ phiên hội thoại Roleplay
    sample_turn_markers = [
        {"role": "student", "start_sec": 0.0, "end_sec": 8.5, "text": "Dạ em chào anh chị, em là chuyên viên tư vấn ạ."},
        {"role": "ai", "start_sec": 8.5, "end_sec": 16.0, "text": "Chào bạn, bên mình đang quan tâm đến khóa học."},
        {"role": "student", "start_sec": 16.0, "end_sec": 26.0, "text": "Dạ vâng, để em chia sẻ lộ trình chi tiết nhé!"}
    ]

    form_data = {
        "session_id": "test_roleplay_session_upload",
        "media_type": "video",
        "step": "6",
        "batch_size": "32",
        "include_timeline_1s": "false",
        "turn_markers": json.dumps(sample_turn_markers, ensure_ascii=False)
    }

    print(f"\n[1] Đang gửi file '{os.path.basename(target_video)}' tới {SERVER_URL}...")
    t0 = time.time()

    with open(target_video, "rb") as f:
        files = {
            "file": (os.path.basename(target_video), f, "video/mp4")
        }
        response = requests.post(SERVER_URL, data=form_data, files=files, timeout=300)

    elapsed = round(time.time() - t0, 2)
    print(f"[2] Thời gian hoàn thành: {elapsed}s | HTTP Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n[3] KẾT QUẢ PHÂN TÍCH THÀNH CÔNG:")
        print(f"  - Session ID: {result.get('session_id')}")
        print(f"  - Thời lượng: {result.get('duration_seconds')}s")
        print(f"  - Tốc độ xử lý: {result.get('performance', {}).get('realtime_multiplier')}")
        print("\n  >> Overall Metrics:")
        metrics = result.get("overall_metrics", {})
        print(f"     + Eye Contact Ratio: {metrics.get('eye_contact_ratio')}")
        print(f"     + Nodding Count: {metrics.get('nodding_count')}")
        print(f"     + Smile Ratio: {metrics.get('smile_ratio')}")
        print(f"     + Speech Rate (WPM): {metrics.get('speech_rate_wpm')}")
        print(f"     + No-face Count: {metrics.get('no_face_count')}")
        print(f"     + No-face Duration: {metrics.get('no_face_duration_sec')}s")
        print(f"     + Vocal Tone: {metrics.get('vocal_tone')}")

        print("\n  >> Turn Evidence Count:", len(result.get("turn_evidence", [])))
        for t in result.get("turn_evidence", []):
            print(f"     [Turn {t.get('turn_index')} - {t.get('role')}]: {t.get('text')} (Eye Contact: {t.get('eye_contact_ratio')}, Nodding: {t.get('nodding_count')})")
    else:
        print(f"[Error] {response.text}")

if __name__ == "__main__":
    test_upload()
