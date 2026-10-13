import json
import urllib.request

def test_api():
    # 1. Fetch random sample from backend
    sample_res = urllib.request.urlopen("http://127.0.0.1:8000/api/sample")
    sample_data = json.loads(sample_res.read())
    true_label = sample_data["label"]
    image_b64 = sample_data["image"]

    # 2. Send predict request
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=json.dumps({"image": image_b64}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    predict_res = urllib.request.urlopen(req)
    result = json.loads(predict_res.read())

    print(f"API Test Result:")
    print(f"  Ground Truth Digit:  {true_label}")
    print(f"  Model Prediction:    {result['prediction']}")
    print(f"  Confidence:          {result['confidence']*100:.2f}%")
    print(f"  Latency:             {result['latency_ms']}ms")
    print(f"  Preview generated:   {'Yes' if len(result['preview']) > 50 else 'No'}")
    assert result['prediction'] == true_label, f"Mismatch: expected {true_label}, got {result['prediction']}"
    print("[PASS] Real-time inference API functioning accurately!")

if __name__ == "__main__":
    test_api()
