#!/usr/bin/env python3
"""
Monitor script that waits for cmann's memory-intensive process to finish,
then launches the bio_llm_only training job.
"""

import subprocess
import time
import os
import signal
import sys

def check_cmann_process():
    """Check if the cmann process is still running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if 'cmann' in line and 'convert_distdf_to_matrix.py' in line:
                return True
        return False
    except subprocess.CalledProcessError:
        return False

def check_memory_usage():
    """Check current memory usage."""
    try:
        result = subprocess.run(
            ['free', '-g'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        if len(lines) >= 2:
            mem_line = lines[1].split()
            if len(mem_line) >= 3:
                total = int(mem_line[1])
                used = int(mem_line[2])
                available = total - used
                return total, used, available
    except:
        pass
    return None, None, None

def launch_training():
    """Launch the training job."""
    print("🚀 Launching training job...")
    
    cmd = [
        'source', '.env', '&&',
        'conda', 'activate', 'verl', '&&',
        'PYTHONPATH=.', 'nohup', 'python', 
        'absolute_zero_reasoner/main_azr_ppo.py',
        '--config-name=bio_llm_only_ppo_trainer',
        'trainer.experiment_name=bio_llm_only',
        'trainer.resume_from_path=False',
        '>', 'training_output.log', '2>&1', '&'
    ]
    
    # Convert to shell command
    shell_cmd = ' '.join(cmd)
    
    try:
        subprocess.run(shell_cmd, shell=True, check=True)
        print("✅ Training job launched successfully!")
        print("📝 Log file: training_output.log")
        print("🔍 Monitor with: tail -f training_output.log")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch training: {e}")

def main():
    print("🔍 Monitoring cmann's memory-intensive process...")
    print("⏳ Waiting for process to finish before launching training...")
    
    # Check initial state
    total, used, available = check_memory_usage()
    if total:
        print(f"💾 Current memory: {used}GB used / {total}GB total ({available}GB available)")
    
    if not check_cmann_process():
        print("✅ cmann process not found! Launching training immediately...")
        launch_training()
        return
    
    # Monitor loop
    check_interval = 30  # seconds
    while True:
        try:
            if not check_cmann_process():
                print("✅ cmann process finished!")
                
                # Wait a bit for memory to be freed
                print("⏳ Waiting 10 seconds for memory cleanup...")
                time.sleep(10)
                
                # Check memory again
                total, used, available = check_memory_usage()
                if total:
                    print(f"💾 Memory after cleanup: {used}GB used / {total}GB total ({available}GB available)")
                
                launch_training()
                break
            
            # Check memory usage
            total, used, available = check_memory_usage()
            if total and available:
                print(f"⏳ cmann still running... Memory: {used}GB used / {total}GB total ({available}GB available)")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main() 