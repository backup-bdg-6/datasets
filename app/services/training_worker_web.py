"""
Training worker service implemented as a web service.
This module provides a Flask wrapper around the training worker for free tier deployment.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, Response

# Configure path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.training_worker import TrainingWorker
from app.config.settings import STORAGE_DIR, IS_FREE_TIER

# Configure logging based on environment
if os.environ.get('RENDER_SERVICE_TYPE', ''):
    # Running on Render.com - use only stream handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Running on Render.com - using stream logging only (no log files)")
else:
    # Local development - can use file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(STORAGE_DIR, 'logs', 'training_worker_web.log'))
        ]
    )
    logger = logging.getLogger(__name__)

# Initialize the training worker
worker = TrainingWorker()

# Global variable to store background threads
background_threads = {}

def create_app():
    """
    Create Flask application for the training worker.
    
    Returns:
        Flask application
    """
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        """
        Health check endpoint.
        """
        memory_info = get_memory_info()
        return jsonify({
            'status': 'ok',
            'timestamp': time.time(),
            'memory': memory_info,
            'active_jobs': list(worker.active_jobs.keys())
        })
    
    @app.route('/training/start', methods=['POST'])
    def start_training():
        """
        Start a training job.
        """
        data = request.json
        training_id = data.get('training_id')
        config = data.get('config')
        
        if not training_id or not config:
            return jsonify({
                'success': False,
                'message': 'Missing training_id or config'
            }), 400
        
        # Start training in a background thread
        thread = threading.Thread(
            target=worker.train_model,
            args=(training_id, config),
            daemon=True
        )
        thread.start()
        
        # Store thread reference
        background_threads[training_id] = thread
        
        return jsonify({
            'success': True,
            'message': f"Training {training_id} started",
            'data': {
                'training_id': training_id,
                'status': 'starting'
            }
        })
    
    @app.route('/training/status/<training_id>', methods=['GET'])
    def training_status(training_id):
        """
        Get training job status.
        """
        # Check if job is active
        if training_id in worker.active_jobs:
            job_info = worker.active_jobs[training_id]
            return jsonify({
                'success': True,
                'data': job_info
            })
        
        # Check task queue for status
        task_info = worker.task_queue.get_task_info(training_id)
        if task_info:
            return jsonify({
                'success': True,
                'data': task_info
            })
        
        return jsonify({
            'success': False,
            'message': f"Training job {training_id} not found"
        }), 404
    
    @app.route('/training/stop/<training_id>', methods=['POST'])
    def stop_training(training_id):
        """
        Stop a training job.
        """
        # Check if job is active
        if training_id in worker.active_jobs:
            worker.stop_training_job(training_id)
            return jsonify({
                'success': True,
                'message': f"Training job {training_id} stopping",
                'data': {
                    'training_id': training_id,
                    'status': 'stopping'
                }
            })
        
        return jsonify({
            'success': False,
            'message': f"Training job {training_id} not found or not active"
        }), 404
    
    @app.route('/export/request', methods=['POST'])
    def request_export():
        """
        Request model export.
        """
        data = request.json
        export_id = data.get('export_id')
        config = data.get('config')
        
        if not export_id or not config:
            return jsonify({
                'success': False,
                'message': 'Missing export_id or config'
            }), 400
        
        # Start export in a background thread
        thread = threading.Thread(
            target=worker.export_model,
            args=(export_id, config),
            daemon=True
        )
        thread.start()
        
        # Store thread reference
        background_threads[export_id] = thread
        
        return jsonify({
            'success': True,
            'message': f"Export {export_id} started",
            'data': {
                'export_id': export_id,
                'status': 'starting'
            }
        })
    
    @app.route('/export/status/<export_id>', methods=['GET'])
    def export_status(export_id):
        """
        Get export job status.
        """
        # Check if job is active
        if export_id in worker.active_jobs:
            job_info = worker.active_jobs[export_id]
            return jsonify({
                'success': True,
                'data': job_info
            })
        
        # Check task queue for status
        task_info = worker.task_queue.get_task_info(export_id)
        if task_info:
            return jsonify({
                'success': True,
                'data': task_info
            })
        
        return jsonify({
            'success': False,
            'message': f"Export job {export_id} not found"
        }), 404
    
    @app.route('/thread/cleanup', methods=['POST'])
    def cleanup_threads():
        """
        Clean up completed background threads.
        """
        completed_count = 0
        for job_id in list(background_threads.keys()):
            thread = background_threads[job_id]
            if not thread.is_alive():
                del background_threads[job_id]
                completed_count += 1
        
        return jsonify({
            'success': True,
            'message': f"Cleaned up {completed_count} completed threads",
            'data': {
                'remaining_threads': len(background_threads)
            }
        })
    
    @app.route('/status', methods=['GET'])
    def worker_status():
        """
        Get overall worker status.
        """
        return jsonify({
            'success': True,
            'data': {
                'active_jobs': list(worker.active_jobs.keys()),
                'active_threads': len(background_threads),
                'memory': get_memory_info()
            }
        })
    
    return app

def get_memory_info() -> Dict[str, Any]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get virtual memory
    virtual_memory = psutil.virtual_memory()
    
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    memory_percent = (memory_usage_mb / MAX_MEMORY_MB) * 100 if MAX_MEMORY_MB else 0
    
    return {
        "used_mb": round(memory_usage_mb, 2),
        "limit_mb": MAX_MEMORY_MB,
        "percent": round(memory_percent, 2),
        "is_critical": memory_percent > 90,
        "system_total_mb": round(virtual_memory.total / (1024 * 1024), 2),
        "system_available_mb": round(virtual_memory.available / (1024 * 1024), 2),
        "system_percent": virtual_memory.percent
    }

# For local development
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8080)
