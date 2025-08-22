#!/usr/bin/env python3
"""
Batch compression script for v1 dataset using tmux sessions
"""

import os
import time
import subprocess
import json
from datetime import datetime

# Configuration
DATA_DIR = "/data2/luzeyi/baseset/PAC/v1_compression_data"
SCRIPT_DIR = "/data2/luzeyi/baseset/PAC/PEARencoding_standalone"
BATCH_SIZE = 8192
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # Available GPUs
CONDA_ENV = "vimm"

# Dataset files
DATASETS = [
    "backup_1GB",
    "book1", 
    "combined_file",
    "enwik9",
    "image",
    "obs_spitzer.trace.fpcgz",
    "sound"
]

def get_file_size(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)

def create_tmux_session(session_name, command):
    """Create a new tmux session and run command"""
    try:
        # Kill existing session if it exists
        subprocess.run(['tmux', 'kill-session', '-t', session_name], 
                      capture_output=True, check=False)
        
        # Create new session
        subprocess.run(['tmux', 'new-session', '-d', '-s', session_name], 
                      check=True)
        
        # Send command to session
        subprocess.run(['tmux', 'send-keys', '-t', session_name, command, 'Enter'], 
                      check=True)
        
        print(f"✅ Started tmux session: {session_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create tmux session {session_name}: {e}")
        return False

def check_tmux_session(session_name):
    """Check if tmux session is still running"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True, check=True)
        return session_name in result.stdout
    except subprocess.CalledProcessError:
        return False

def get_compression_progress(dataset_name):
    """Monitor compression progress and calculate stats"""
    input_file = os.path.join(DATA_DIR, dataset_name)
    compressed_file = f"{dataset_name}_16_256_4096_bs{BATCH_SIZE}_1_seq1.compressed.combined"
    
    if not os.path.exists(input_file):
        return None
        
    original_size = get_file_size(input_file)
    
    # Check if compression is complete
    if os.path.exists(compressed_file):
        compressed_size = get_file_size(compressed_file)
        compression_ratio = original_size / compressed_size
        
        return {
            'status': 'completed',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'file_exists': True
        }
    
    # Check if temporary directory exists (compression in progress)
    temp_dir = f"{dataset_name}_16_256_4096_bs{BATCH_SIZE}_1_seq1_temp"
    if os.path.exists(temp_dir):
        temp_size = sum(os.path.getsize(os.path.join(temp_dir, f)) 
                       for f in os.listdir(temp_dir) 
                       if os.path.isfile(os.path.join(temp_dir, f)))
        
        return {
            'status': 'in_progress',
            'original_size': original_size,
            'temp_size': temp_size,
            'progress_estimate': min(temp_size / (original_size * 0.3) * 100, 99),  # Rough estimate
            'file_exists': True
        }
    
    return {
        'status': 'not_started',
        'original_size': original_size,
        'file_exists': True
    }

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def calculate_speed(original_size, elapsed_time):
    """Calculate compression speed in KB/s"""
    if elapsed_time > 0:
        return (original_size / 1024) / elapsed_time
    return 0

def start_compression_jobs():
    """Start compression jobs for all datasets"""
    print("🚀 Starting batch compression for v1 dataset...")
    print(f"📊 Found {len(DATASETS)} datasets to compress")
    print(f"⚙️  Batch size: {BATCH_SIZE}")
    print("=" * 60)
    
    job_info = {}
    
    for i, dataset in enumerate(DATASETS):
        input_file = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(input_file):
            print(f"⚠️  File not found: {dataset}")
            continue
            
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        session_name = f"compress_{dataset}"
        
        # Build compression command
        command = (
            f"cd {SCRIPT_DIR} && "
            f"source /data2/luzeyi/anaconda3/etc/profile.d/conda.sh && "
            f"conda activate {CONDA_ENV} && "
            f"python PEARencodingdic.py "
            f"--input_dir {input_file} "
            f"--prefix {dataset} "
            f"--gpu_id {gpu_id} "
            f"--batch_size {BATCH_SIZE}"
        )
        
        if create_tmux_session(session_name, command):
            job_info[dataset] = {
                'session_name': session_name,
                'gpu_id': gpu_id,
                'start_time': time.time(),
                'original_size': get_file_size(input_file)
            }
            
            print(f"📂 {dataset}: GPU {gpu_id}, Size: {format_size(get_file_size(input_file))}")
        
        # Small delay between job starts
        time.sleep(2)
    
    print("=" * 60)
    print(f"✅ Started {len(job_info)} compression jobs")
    
    # Save job info
    with open('compression_jobs.json', 'w') as f:
        json.dump(job_info, f, indent=2)
    
    return job_info

def monitor_progress():
    """Monitor compression progress and display results"""
    print("\n🔍 Monitoring compression progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        # Load job info
        if not os.path.exists('compression_jobs.json'):
            print("❌ No job info found. Please start jobs first.")
            return
            
        with open('compression_jobs.json', 'r') as f:
            job_info = json.load(f)
        
        completed_jobs = set()
        
        while len(completed_jobs) < len(job_info):
            os.system('clear')  # Clear screen
            print("🏗️  PEAR Compression Progress Monitor")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            for dataset, job in job_info.items():
                if dataset in completed_jobs:
                    continue
                    
                session_name = job['session_name']
                start_time = job['start_time']
                elapsed_time = time.time() - start_time
                
                # Check if tmux session is still running
                session_running = check_tmux_session(session_name)
                
                # Get compression progress
                progress = get_compression_progress(dataset)
                
                if progress:
                    status = progress['status']
                    original_size = progress['original_size']
                    
                    if status == 'completed':
                        compressed_size = progress['compressed_size']
                        ratio = progress['compression_ratio']
                        speed = calculate_speed(original_size, elapsed_time)
                        
                        print(f"✅ {dataset:<20} COMPLETED")
                        print(f"   📏 Original: {format_size(original_size):<12} "
                              f"Compressed: {format_size(compressed_size):<12}")
                        print(f"   📈 Ratio: {ratio:.2f}x "
                              f"Speed: {speed:.1f} KB/s "
                              f"Time: {elapsed_time/60:.1f}m")
                        print()
                        
                        completed_jobs.add(dataset)
                        
                    elif status == 'in_progress':
                        temp_size = progress.get('temp_size', 0)
                        progress_pct = progress.get('progress_estimate', 0)
                        speed = calculate_speed(original_size * progress_pct / 100, elapsed_time) if elapsed_time > 0 else 0
                        
                        print(f"🔄 {dataset:<20} IN PROGRESS ({progress_pct:.1f}%)")
                        print(f"   📏 Original: {format_size(original_size):<12} "
                              f"Temp: {format_size(temp_size):<12}")
                        print(f"   ⚡ Speed: {speed:.1f} KB/s "
                              f"Time: {elapsed_time/60:.1f}m "
                              f"GPU: {job['gpu_id']}")
                        print()
                        
                    else:
                        print(f"⏳ {dataset:<20} WAITING")
                        print(f"   📏 Size: {format_size(original_size):<12} "
                              f"GPU: {job['gpu_id']}")
                        print()
                
                # Check if session died unexpectedly
                if not session_running and dataset not in completed_jobs:
                    print(f"❌ {dataset:<20} SESSION ENDED (check for errors)")
                    print(f"   💡 Use: tmux attach -t {session_name}")
                    print()
            
            print("=" * 80)
            print(f"📊 Progress: {len(completed_jobs)}/{len(job_info)} completed")
            
            if len(completed_jobs) < len(job_info):
                time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")

def show_results():
    """Show final compression results"""
    print("\n📋 Final Compression Results")
    print("=" * 80)
    
    total_original = 0
    total_compressed = 0
    
    for dataset in DATASETS:
        progress = get_compression_progress(dataset)
        if progress and progress['status'] == 'completed':
            original_size = progress['original_size']
            compressed_size = progress['compressed_size']
            ratio = progress['compression_ratio']
            
            total_original += original_size
            total_compressed += compressed_size
            
            print(f"{dataset:<25} {format_size(original_size):<12} → "
                  f"{format_size(compressed_size):<12} ({ratio:.2f}x)")
    
    if total_compressed > 0:
        overall_ratio = total_original / total_compressed
        print("-" * 80)
        print(f"{'TOTAL':<25} {format_size(total_original):<12} → "
              f"{format_size(total_compressed):<12} ({overall_ratio:.2f}x)")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_compress.py start     - Start all compression jobs")
        print("  python batch_compress.py monitor   - Monitor progress")
        print("  python batch_compress.py results   - Show results")
        return
    
    command = sys.argv[1]
    
    if command == "start":
        start_compression_jobs()
    elif command == "monitor":
        monitor_progress()
    elif command == "results":
        show_results()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()