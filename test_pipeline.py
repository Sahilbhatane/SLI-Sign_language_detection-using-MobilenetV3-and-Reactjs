"""
End-to-End Pipeline Test Orchestrator
Tests the complete Sign Language Recognition system:
- Backend API (FastAPI)
- Frontend build (React)
- Model evaluation
- API endpoint tests
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import signal
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import requests


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PipelineOrchestrator:
    """Orchestrates end-to-end testing of the SLI pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.results: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'backend': {},
            'frontend': {},
            'tests': {},
            'summary': {}
        }
        
    def print_header(self, text: str):
        """Print formatted section header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")
    
    def print_info(self, text: str):
        """Print info message"""
        print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")
    
    async def check_url_ready(self, url: str, timeout: int = 60, interval: float = 1.0) -> bool:
        """
        Check if URL is reachable
        
        Args:
            url: URL to check
            timeout: Maximum time to wait (seconds)
            interval: Check interval (seconds)
        
        Returns:
            True if URL is reachable, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code < 500:
                    return True
            except requests.exceptions.RequestException:
                pass
            await asyncio.sleep(interval)
        return False
    
    def start_backend(self) -> bool:
        """
        Start FastAPI backend server
        
        Returns:
            True if started successfully
        """
        self.print_info("Starting FastAPI backend...")
        
        try:
            # Check if model exists
            model_path = self.project_root / "backend" / "model_v2.onnx"
            if not model_path.exists():
                self.print_warning(f"Model not found: {model_path}")
                self.print_warning("Backend may fail to start. Consider training the model first.")
            
            # Start uvicorn
            backend_dir = self.project_root / "backend"
            
            # Use Python executable from current environment
            python_exe = sys.executable
            
            # Start backend process
            if platform.system() == "Windows":
                self.backend_process = subprocess.Popen(
                    [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
                    cwd=backend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                self.backend_process = subprocess.Popen(
                    [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
                    cwd=backend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            self.print_success(f"Backend process started (PID: {self.backend_process.pid})")
            self.results['backend']['pid'] = self.backend_process.pid
            self.results['backend']['status'] = 'started'
            return True
            
        except Exception as e:
            self.print_error(f"Failed to start backend: {e}")
            self.results['backend']['status'] = 'failed'
            self.results['backend']['error'] = str(e)
            return False
    
    def start_frontend(self) -> bool:
        """
        Build and start React frontend
        
        Returns:
            True if started successfully
        """
        self.print_info("Building and starting React frontend...")
        
        try:
            frontend_dir = self.project_root / "frontend"
            
            # Check if node_modules exists
            if not (frontend_dir / "node_modules").exists():
                self.print_warning("node_modules not found. Run 'npm install' first.")
                self.print_info("Skipping frontend for now...")
                self.results['frontend']['status'] = 'skipped'
                return True
            
            # Build frontend
            self.print_info("Building frontend (this may take a minute)...")
            build_result = subprocess.run(
                ["npm", "run", "build"],
                cwd=frontend_dir,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if build_result.returncode != 0:
                self.print_error(f"Frontend build failed: {build_result.stderr[:200]}")
                self.results['frontend']['status'] = 'build_failed'
                self.results['frontend']['error'] = build_result.stderr[:500]
                return False
            
            self.print_success("Frontend built successfully")
            
            # Start frontend server with npx serve
            dist_dir = frontend_dir / "dist"
            if not dist_dir.exists():
                self.print_error("Build output directory not found")
                return False
            
            self.print_info("Starting frontend server...")
            
            if platform.system() == "Windows":
                self.frontend_process = subprocess.Popen(
                    ["npx", "serve", "-s", "dist", "-l", "3000"],
                    cwd=frontend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                self.frontend_process = subprocess.Popen(
                    ["npx", "serve", "-s", "dist", "-l", "3000"],
                    cwd=frontend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            self.print_success(f"Frontend process started (PID: {self.frontend_process.pid})")
            self.results['frontend']['pid'] = self.frontend_process.pid
            self.results['frontend']['status'] = 'started'
            return True
            
        except subprocess.TimeoutExpired:
            self.print_error("Frontend build timed out")
            self.results['frontend']['status'] = 'timeout'
            return False
        except FileNotFoundError:
            self.print_warning("npm or npx not found. Skipping frontend.")
            self.results['frontend']['status'] = 'skipped'
            return True
        except Exception as e:
            self.print_error(f"Failed to start frontend: {e}")
            self.results['frontend']['status'] = 'failed'
            self.results['frontend']['error'] = str(e)
            return False
    
    async def wait_for_services(self) -> bool:
        """
        Wait for backend and frontend to be ready
        
        Returns:
            True if all services are ready
        """
        self.print_header("Waiting for Services to Start")
        
        # Wait for backend
        self.print_info("Checking backend health...")
        backend_ready = await self.check_url_ready("http://localhost:8000/health", timeout=30)
        
        if backend_ready:
            self.print_success("Backend is ready (http://localhost:8000)")
            self.results['backend']['ready'] = True
        else:
            self.print_error("Backend failed to start within timeout")
            self.results['backend']['ready'] = False
            return False
        
        # Wait for frontend (if started)
        if self.frontend_process:
            self.print_info("Checking frontend...")
            frontend_ready = await self.check_url_ready("http://localhost:3000", timeout=20)
            
            if frontend_ready:
                self.print_success("Frontend is ready (http://localhost:3000)")
                self.results['frontend']['ready'] = True
            else:
                self.print_warning("Frontend not reachable (may still be starting)")
                self.results['frontend']['ready'] = False
        
        return True
    
    def test_backend_endpoints(self) -> Dict[str, Any]:
        """
        Test all backend API endpoints
        
        Returns:
            Test results dictionary
        """
        self.print_header("Testing Backend API Endpoints")
        
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        base_url = "http://localhost:8000"
        
        # Test 1: Root endpoint
        test_name = "GET /"
        results['total'] += 1
        try:
            start = time.time()
            response = requests.get(f"{base_url}/", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200 and 'message' in response.json():
                self.print_success(f"{test_name} - {latency:.0f}ms")
                results['passed'] += 1
                results['tests'].append({'name': test_name, 'status': 'pass', 'latency_ms': latency})
            else:
                self.print_error(f"{test_name} - Unexpected response")
                results['failed'] += 1
                results['tests'].append({'name': test_name, 'status': 'fail', 'error': 'Unexpected response'})
        except Exception as e:
            self.print_error(f"{test_name} - {e}")
            results['failed'] += 1
            results['tests'].append({'name': test_name, 'status': 'fail', 'error': str(e)})
        
        # Test 2: Health check
        test_name = "GET /health"
        results['total'] += 1
        try:
            start = time.time()
            response = requests.get(f"{base_url}/health", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"{test_name} - {latency:.0f}ms (model_loaded: {data.get('model_loaded')})")
                results['passed'] += 1
                results['tests'].append({
                    'name': test_name,
                    'status': 'pass',
                    'latency_ms': latency,
                    'model_loaded': data.get('model_loaded')
                })
            else:
                self.print_error(f"{test_name} - Status {response.status_code}")
                results['failed'] += 1
                results['tests'].append({'name': test_name, 'status': 'fail', 'status_code': response.status_code})
        except Exception as e:
            self.print_error(f"{test_name} - {e}")
            results['failed'] += 1
            results['tests'].append({'name': test_name, 'status': 'fail', 'error': str(e)})
        
        # Test 3: Get classes
        test_name = "GET /classes"
        results['total'] += 1
        try:
            start = time.time()
            response = requests.get(f"{base_url}/classes", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                num_classes = data.get('count', 0)
                self.print_success(f"{test_name} - {latency:.0f}ms ({num_classes} classes)")
                results['passed'] += 1
                results['tests'].append({
                    'name': test_name,
                    'status': 'pass',
                    'latency_ms': latency,
                    'num_classes': num_classes
                })
            else:
                self.print_error(f"{test_name} - Status {response.status_code}")
                results['failed'] += 1
                results['tests'].append({'name': test_name, 'status': 'fail', 'status_code': response.status_code})
        except Exception as e:
            self.print_error(f"{test_name} - {e}")
            results['failed'] += 1
            results['tests'].append({'name': test_name, 'status': 'fail', 'error': str(e)})
        
        # Test 4: Predict with dummy image
        test_name = "POST /predict (dummy image)"
        results['total'] += 1
        try:
            # Create a small dummy base64 image
            import base64
            from PIL import Image
            import io
            
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            payload = {
                "image": b64_image,
                "top_k": 5
            }
            
            start = time.time()
            response = requests.post(f"{base_url}/predict", json=payload, timeout=10)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', 'N/A')
                confidence = data.get('confidence', 0)
                self.print_success(f"{test_name} - {latency:.0f}ms (pred: {prediction}, conf: {confidence:.1f}%)")
                results['passed'] += 1
                results['tests'].append({
                    'name': test_name,
                    'status': 'pass',
                    'latency_ms': latency,
                    'prediction': prediction,
                    'confidence': confidence
                })
            else:
                self.print_error(f"{test_name} - Status {response.status_code}")
                results['failed'] += 1
                results['tests'].append({'name': test_name, 'status': 'fail', 'status_code': response.status_code})
        except Exception as e:
            self.print_error(f"{test_name} - {e}")
            results['failed'] += 1
            results['tests'].append({'name': test_name, 'status': 'fail', 'error': str(e)})
        
        return results
    
    def run_model_evaluation(self) -> Dict[str, Any]:
        """
        Run model evaluation if model exists
        
        Returns:
            Evaluation results dictionary
        """
        self.print_header("Running Model Evaluation")
        
        results = {
            'status': 'skipped',
            'reason': 'Model or data not found'
        }
        
        model_path = self.project_root / "backend" / "model_v2.onnx"
        data_path = self.project_root / "data"
        
        if not model_path.exists():
            self.print_warning(f"Model not found: {model_path}")
            self.print_info("Skipping model evaluation. Train the model first.")
            return results
        
        if not data_path.exists():
            self.print_warning(f"Data directory not found: {data_path}")
            return results
        
        # Check if data has subdirectories (classes)
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            self.print_warning("No class directories found in data/")
            return results
        
        self.print_info(f"Running evaluation on {len(class_dirs)} classes...")
        
        try:
            # Run evaluate_model.py as a subprocess with limited samples
            eval_script = self.project_root / "ML" / "evaluate_model.py"
            
            result = subprocess.run(
                [
                    sys.executable,
                    str(eval_script),
                    "--model", str(model_path),
                    "--data", str(data_path),
                    "--out", "evaluation_test",
                    "--max-samples", "5"  # Limit samples for quick test
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.print_success("Model evaluation completed")
                
                # Try to parse accuracy from output
                output_lines = result.stdout.split('\n')
                accuracy = None
                for line in output_lines:
                    if 'Accuracy:' in line:
                        try:
                            accuracy = float(line.split(':')[1].strip().rstrip('%'))
                        except:
                            pass
                
                results['status'] = 'completed'
                results['accuracy'] = accuracy
                if accuracy:
                    self.print_info(f"Accuracy: {accuracy:.2f}%")
            else:
                self.print_error(f"Evaluation failed: {result.stderr[:200]}")
                results['status'] = 'failed'
                results['error'] = result.stderr[:500]
        
        except subprocess.TimeoutExpired:
            self.print_error("Evaluation timed out")
            results['status'] = 'timeout'
        except Exception as e:
            self.print_error(f"Evaluation error: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def shutdown_services(self):
        """Gracefully shutdown backend and frontend"""
        self.print_header("Shutting Down Services")
        
        # Shutdown backend
        if self.backend_process:
            self.print_info("Stopping backend...")
            try:
                if platform.system() == "Windows":
                    self.backend_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.backend_process.pid), signal.SIGTERM)
                
                self.backend_process.wait(timeout=5)
                self.print_success("Backend stopped")
            except subprocess.TimeoutExpired:
                self.print_warning("Backend did not stop gracefully, forcing...")
                self.backend_process.kill()
            except Exception as e:
                self.print_error(f"Error stopping backend: {e}")
        
        # Shutdown frontend
        if self.frontend_process:
            self.print_info("Stopping frontend...")
            try:
                if platform.system() == "Windows":
                    self.frontend_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.frontend_process.pid), signal.SIGTERM)
                
                self.frontend_process.wait(timeout=5)
                self.print_success("Frontend stopped")
            except subprocess.TimeoutExpired:
                self.print_warning("Frontend did not stop gracefully, forcing...")
                self.frontend_process.kill()
            except Exception as e:
                self.print_error(f"Error stopping frontend: {e}")
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        # Backend tests
        backend_tests = self.results.get('tests', {}).get('backend', {})
        if backend_tests:
            total = backend_tests.get('total', 0)
            passed = backend_tests.get('passed', 0)
            failed = backend_tests.get('failed', 0)
            
            print(f"{Colors.BOLD}Backend API Tests:{Colors.ENDC}")
            print(f"  Total:  {total}")
            print(f"  Passed: {Colors.GREEN}{passed}{Colors.ENDC}")
            print(f"  Failed: {Colors.RED}{failed}{Colors.ENDC}")
            
            if total > 0:
                success_rate = (passed / total) * 100
                print(f"  Success Rate: {success_rate:.1f}%")
            print()
        
        # Model evaluation
        evaluation = self.results.get('tests', {}).get('evaluation', {})
        if evaluation:
            print(f"{Colors.BOLD}Model Evaluation:{Colors.ENDC}")
            print(f"  Status: {evaluation.get('status', 'N/A')}")
            if 'accuracy' in evaluation and evaluation['accuracy']:
                print(f"  Accuracy: {evaluation['accuracy']:.2f}%")
            print()
        
        # Overall status
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.results['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        print(f"{Colors.BOLD}Overall:{Colors.ENDC}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Start: {start_time.strftime('%H:%M:%S')}")
        print(f"  End: {end_time.strftime('%H:%M:%S')}")
        
        # Save results to JSON
        results_file = self.project_root / "pipeline_test_results.json"
        self.results['end_time'] = end_time.isoformat()
        self.results['duration_seconds'] = duration
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{Colors.CYAN}Results saved to: {results_file}{Colors.ENDC}")
    
    async def run(self):
        """Main orchestration flow"""
        try:
            self.print_header("Sign Language Recognition - End-to-End Pipeline Test")
            
            # Step 1: Start Backend
            if not self.start_backend():
                self.print_error("Failed to start backend. Aborting.")
                return
            
            # Step 2: Start Frontend
            self.start_frontend()  # Continue even if frontend fails
            
            # Step 3: Wait for services
            await asyncio.sleep(2)  # Give processes time to initialize
            if not await self.wait_for_services():
                self.print_error("Services failed to start. Aborting tests.")
                return
            
            # Step 4: Run backend tests
            backend_results = self.test_backend_endpoints()
            self.results['tests']['backend'] = backend_results
            
            # Step 5: Run model evaluation (optional)
            evaluation_results = self.run_model_evaluation()
            self.results['tests']['evaluation'] = evaluation_results
            
            # Step 6: Print summary
            self.print_summary()
            
            # Determine overall success
            backend_passed = backend_results.get('passed', 0)
            backend_total = backend_results.get('total', 0)
            
            if backend_total > 0 and backend_passed == backend_total:
                self.print_header("✓ ALL TESTS PASSED")
            else:
                self.print_header("✗ SOME TESTS FAILED")
        
        except KeyboardInterrupt:
            self.print_warning("\nTests interrupted by user")
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always shutdown services
            self.shutdown_services()


async def main():
    """Main entry point"""
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    # Run the async orchestrator
    asyncio.run(main())
