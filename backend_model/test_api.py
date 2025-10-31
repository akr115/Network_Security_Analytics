"""
Test script for OCSVM Intrusion Detection API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_get_features():
    """Test get features endpoint"""
    print("\n" + "="*60)
    print("Testing Get Features")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total Features: {data['feature_count']}")
    print(f"First 10 Features: {data['features'][:10]}")
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    # Sample data (you'll need to provide actual feature values)
    sample_data = {
        "data": {
            "Destination Port": 80,
            "Flow Duration": 120000,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 8,
            "Total Length of Fwd Packets": 5000,
            "Total Length of Bwd Packets": 3000,
            "Fwd Packet Length Max": 1500,
            "Fwd Packet Length Min": 60,
            "Fwd Packet Length Mean": 500.0,
            "Fwd Packet Length Std": 100.0,
            # Add more features as needed
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Is Anomaly: {result['is_anomaly']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    # Sample batch data
    batch_data = {
        "flows": [
            {
                "Destination Port": 80,
                "Flow Duration": 120000,
                "Total Fwd Packets": 10,
                "Total Backward Packets": 8,
            },
            {
                "Destination Port": 443,
                "Flow Duration": 90000,
                "Total Fwd Packets": 15,
                "Total Backward Packets": 12,
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Flows: {result['total_flows']}")
        print(f"Benign Count: {result['benign_count']}")
        print(f"Attack Count: {result['attack_count']}")
        print(f"\nPredictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  Flow {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.4f})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("OCSVM API Test Suite")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Get Features", test_get_features),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n {test_name} failed with error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results.items():
        status = " PASSED" if passed else " FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n Error: Could not connect to API")
        print("Make sure the API is running at", BASE_URL)
        print("\nStart the API with:")
        print("  python main.py")
        print("  or")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
