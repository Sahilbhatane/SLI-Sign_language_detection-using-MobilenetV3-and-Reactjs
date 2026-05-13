"""
Integration tests for FastAPI routes using an in-process ASGI client.
Runs with a fake model, so no ONNX model file is required.
Run from backend/: ..\\venv\\Scripts\\python.exe -m unittest test_integration_api -v
"""

from __future__ import annotations

import asyncio
import base64
import unittest

import httpx

import main

VALID_IMAGE_B64 = base64.b64encode(b"fake-image-bytes").decode("ascii")


class FakeModel:
    def __init__(self, top_conf: float = 0.92) -> None:
        self.top_conf = max(0.0, min(1.0, float(top_conf)))
        self.model_path = "fake.onnx"
        self.class_labels = ["hello", "thanks", "yes"]
        self.num_classes = len(self.class_labels)

    def get_model_info(self):
        return {
            "num_classes": self.num_classes,
            "input_shape": [1, 224, 224, 3],
            "input_name": "input",
            "providers": ["CPUExecutionProvider"],
            "model_path": self.model_path,
            "classes": self.class_labels,
        }

    def predict_from_base64(self, image_b64: str, top_k: int = 5):
        rest = max(0.0, 1.0 - self.top_conf)
        preds = [
            {
                "rank": 1,
                "class": self.class_labels[0],
                "confidence": self.top_conf,
                "confidence_percent": self.top_conf * 100.0,
            },
            {
                "rank": 2,
                "class": self.class_labels[1],
                "confidence": rest,
                "confidence_percent": rest * 100.0,
            },
            {
                "rank": 3,
                "class": self.class_labels[2],
                "confidence": 0.01,
                "confidence_percent": 1.0,
            },
        ]
        limit = max(1, min(int(top_k), len(preds)))
        return preds[:limit]


class TestApiIntegration(unittest.TestCase):
    def setUp(self) -> None:
        main.model_instance = FakeModel()

    def tearDown(self) -> None:
        main.model_instance = None

    def _request(self, method: str, path: str, json_body=None):
        async def _call():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, json=json_body)

        return asyncio.run(_call())

    def test_root(self) -> None:
        response = self._request("GET", "/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("endpoints", data)
        self.assertIn("predict", data["endpoints"])

    def test_health(self) -> None:
        response = self._request("GET", "/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["num_classes"], 3)

    def test_classes(self) -> None:
        response = self._request("GET", "/classes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 3)
        self.assertEqual(len(data["classes"]), 3)

    def test_model_info(self) -> None:
        response = self._request("GET", "/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_info", data)
        info = data["model_info"]
        self.assertEqual(info["num_classes"], 3)
        self.assertEqual(info["input_name"], "input")

    def test_predict_accepts_confident_top1(self) -> None:
        payload = {
            "image": f"data:image/png;base64,{VALID_IMAGE_B64}",
            "top_k": 2,
            "min_confidence": 0.6,
        }
        response = self._request("POST", "/predict", json_body=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "hello")
        self.assertGreater(data["confidence"], 0.0)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertIn("class", data["predictions"][0])

    def test_predict_below_min_confidence(self) -> None:
        main.model_instance = FakeModel(top_conf=0.4)
        payload = {
            "image": VALID_IMAGE_B64,
            "top_k": 2,
            "min_confidence": 0.6,
        }
        response = self._request("POST", "/predict", json_body=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "Detecting...")
        self.assertEqual(data["confidence"], 0.0)
        self.assertEqual(data["min_confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
