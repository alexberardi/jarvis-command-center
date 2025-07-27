#!/usr/bin/env python3
"""
Script to run only the real LLM API tests from the test suite

This script identifies and runs tests that make real HTTP calls to the LLM proxy API,
collecting comprehensive metrics on performance, accuracy, and response quality.

Usage:
    python run_real_llm_tests.py
    python run_real_llm_tests.py --save-metrics
    python run_real_llm_tests.py --output-file=my_metrics.json
"""
import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load .env file at module import
load_env_file()

def check_llm_api_url():
    """Check if LLM API URL is configured"""
    api_url = os.getenv("JARVIS_LLM_PROXY_API_URL")
    if not api_url:
        print("⚠️  JARVIS_LLM_PROXY_API_URL environment variable not set!")
        print("   This is required for running tests, but not for listing them.")
        print("   Example: export JARVIS_LLM_PROXY_API_URL=http://10.0.0.69:8000")
        return False
    
    print(f"✅ LLM API URL configured: {api_url}")
    return True

def get_real_llm_tests():
    """Get list of tests that make real LLM API calls"""
    real_llm_tests = [
        # From test_llm_integration.py
        "tests/test_llm_integration.py::TestTranscriptionCleanup::test_transcription_cleanup_with_filler_words",
        "tests/test_llm_integration.py::TestTranscriptionCleanup::test_transcription_cleanup_performance_comparison",
        "tests/test_llm_integration.py::TestTranscriptionCleanup::test_clean_already_clean_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_basic_light_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_temperature_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_timer_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_music_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_missing_room_context",
        "tests/test_llm_integration.py::TestLLMIntegration::test_ambiguous_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_unrecognized_commands",
        "tests/test_llm_integration.py::TestLLMIntegration::test_natural_language_variations",
        "tests/test_llm_integration.py::TestLLMIntegration::test_complex_scenarios",
        "tests/test_llm_integration.py::TestLLMIntegration::test_multiple_commands",
        "tests/test_llm_integration.py::TestLLMPerformance::test_response_time",
    ]
    
    return real_llm_tests

def run_pytest_with_metrics(test_paths: List[str], output_file: str = None) -> Dict[str, Any]:
    """Run pytest with metrics collection"""
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short"
    ]
    
    # Add test paths
    cmd.extend(test_paths)
    
    print(f"🚀 Running real LLM API tests...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    # Run the tests
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()
    
    # Parse results
    test_results = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "tests_run": len(test_paths),
        "test_paths": test_paths
    }
    
    # Note: JSON report not available without pytest-json-report plugin
    test_results["json_report_note"] = "Install pytest-json-report for detailed JSON reports"
    
    return test_results

def extract_metrics_from_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract metrics from pytest output"""
    metrics = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "response_times": [],
        "token_usage": [],
        "costs": [],
        "success_rate": 0.0
    }
    
    lines = stdout.split('\n')
    
    for line in lines:
        # Count test results
        if "PASSED" in line:
            metrics["passed"] += 1
            metrics["total_tests"] += 1
        elif "FAILED" in line:
            metrics["failed"] += 1
            metrics["total_tests"] += 1
        elif "SKIPPED" in line:
            metrics["skipped"] += 1
            metrics["total_tests"] += 1
        elif "ERROR" in line:
            metrics["errors"] += 1
            metrics["total_tests"] += 1
        
        # Extract response times
        if "Response time:" in line or "response time:" in line:
            try:
                time_str = line.split(":")[-1].strip()
                if "ms" in time_str:
                    time_ms = float(time_str.replace("ms", "").strip())
                    metrics["response_times"].append(time_ms)
                elif "s" in time_str:
                    time_s = float(time_str.replace("s", "").strip())
                    metrics["response_times"].append(time_s * 1000)
            except:
                pass
        
        # Extract token usage
        if "tokens:" in line.lower() or "Tokens:" in line:
            try:
                token_str = line.split(":")[-1].strip()
                tokens = int(token_str.replace(",", ""))
                metrics["token_usage"].append(tokens)
            except:
                pass
        
        # Extract costs
        if "cost:" in line.lower() or "Cost:" in line:
            try:
                cost_str = line.split(":")[-1].strip()
                if "$" in cost_str:
                    cost = float(cost_str.replace("$", "").strip())
                    metrics["costs"].append(cost)
            except:
                pass
    
    # Calculate success rate
    if metrics["total_tests"] > 0:
        metrics["success_rate"] = (metrics["passed"] / metrics["total_tests"]) * 100
    
    return metrics

def print_metrics_summary(metrics: Dict[str, Any]):
    """Print comprehensive metrics summary"""
    print(f"\n{'='*80}")
    print("📊 REAL LLM API TESTS METRICS SUMMARY")
    print(f"{'='*80}")
    
    print(f"📈 Test Results:")
    print(f"   Total Tests: {metrics['total_tests']}")
    print(f"   Passed: {metrics['passed']}")
    print(f"   Failed: {metrics['failed']}")
    print(f"   Skipped: {metrics['skipped']}")
    print(f"   Errors: {metrics['errors']}")
    print(f"   Success Rate: {metrics['success_rate']:.1f}%")
    
    if metrics['response_times']:
        print(f"\n⏱️  Response Time Metrics:")
        print(f"   Average: {sum(metrics['response_times']) / len(metrics['response_times']):.2f}ms")
        print(f"   Min: {min(metrics['response_times']):.2f}ms")
        print(f"   Max: {max(metrics['response_times']):.2f}ms")
        print(f"   Total: {len(metrics['response_times'])} measurements")
    
    if metrics['token_usage']:
        print(f"\n🎯 Token Usage Metrics:")
        print(f"   Average: {sum(metrics['token_usage']) / len(metrics['token_usage']):.0f} tokens")
        print(f"   Total: {sum(metrics['token_usage']):,} tokens")
        print(f"   Min: {min(metrics['token_usage'])} tokens")
        print(f"   Max: {max(metrics['token_usage'])} tokens")
    
    if metrics['costs']:
        print(f"\n💰 Cost Metrics:")
        print(f"   Average: ${sum(metrics['costs']) / len(metrics['costs']):.4f}")
        print(f"   Total: ${sum(metrics['costs']):.4f}")
        print(f"   Min: ${min(metrics['costs']):.4f}")
        print(f"   Max: ${max(metrics['costs']):.4f}")

def save_metrics_to_file(metrics: Dict[str, Any], filename: str):
    """Save metrics to JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {filename}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run real LLM API tests with metrics collection")
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to JSON file"
    )
    parser.add_argument(
        "--output-file",
        default="real_llm_test_metrics.json",
        help="Output file for metrics (default: real_llm_test_metrics.json)"
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all real LLM API tests without running them"
    )
    
    args = parser.parse_args()
    
    # Check if LLM API URL is configured (only required for running tests, not listing)
    api_url_configured = check_llm_api_url()
    
    # Get real LLM tests
    real_llm_tests = get_real_llm_tests()
    
    if args.list_tests:
        print(f"\n📋 Real LLM API Tests ({len(real_llm_tests)} tests):")
        for i, test in enumerate(real_llm_tests, 1):
            print(f"   {i:2d}. {test}")
        return
    
    # For running tests, API URL is required
    if not api_url_configured:
        print("\n❌ Cannot run tests without JARVIS_LLM_PROXY_API_URL configured!")
        sys.exit(1)
    
    print(f"\n🎯 Found {len(real_llm_tests)} real LLM API tests")
    
    # Run the tests
    test_results = run_pytest_with_metrics(real_llm_tests, args.output_file)
    
    # Extract metrics from output
    metrics = extract_metrics_from_output(test_results["stdout"], test_results["stderr"])
    
    # Add test results to metrics
    metrics.update({
        "test_results": test_results,
        "llm_api_url": os.getenv("JARVIS_LLM_PROXY_API_URL"),
        "timestamp": datetime.now().isoformat()
    })
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save metrics if requested
    if args.save_metrics:
        save_metrics_to_file(metrics, args.output_file)
    
    # Print test output
    print(f"\n{'='*80}")
    print("📄 TEST OUTPUT")
    print(f"{'='*80}")
    print(test_results["stdout"])
    
    if test_results["stderr"]:
        print(f"\n{'='*80}")
        print("⚠️  ERROR OUTPUT")
        print(f"{'='*80}")
        print(test_results["stderr"])
    
    # Exit with appropriate code
    sys.exit(test_results["return_code"])

if __name__ == "__main__":
    main() 