import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    print("Testing /embed")
    embed_response = post("/embed", {"text": "This is a test sentence."})
    print(json.dumps(embed_response, indent=2))

    print("\nTesting /similarity")
    similarity_response = post(
        "/similarity",
        {"text1": "FastAPI is fast.", "text2": "FastAPI is a web framework."},
    )
    print(json.dumps(similarity_response, indent=2))
