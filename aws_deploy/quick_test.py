import requests
import json

API_URL = "https://2ssx8bnfcf.execute-api.us-east-1.amazonaws.com/predict"

def test_api():
    payload = {"text": "after another , most of which involve precocious kids getting the better of obnoxious adults"}
    headers = {"Content-Type": "application/json"}
    
    print(f"Testing API: {API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\n" + "="*60)
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("\nResponse Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
