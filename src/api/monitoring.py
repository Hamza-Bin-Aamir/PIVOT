"""System resource monitoring for CPU, memory, disk, and GPU.

This module provides utilities for monitoring system resources including
CPU usage, memory usage, disk space, and GPU metrics if available.
"""

import platform
from datetime import datetime, timezone
from typing import Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ResourceMonitor:
    """Monitor system resources."""

    def get_cpu_info(self) -> dict[str, Any]:
        """Get CPU information and usage.

        Returns:
            CPU metrics including usage percentage per core and total
        """
        if not PSUTIL_AVAILABLE:
            return {'available': False, 'error': 'psutil not installed'}

        return {
            'available': True,
            'count': psutil.cpu_count(logical=True),
            'percent': psutil.cpu_percent(interval=0.1, percpu=False),
            'percent_per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
            'frequency': {
                'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            } if psutil.cpu_freq() else None,
        }

    def get_memory_info(self) -> dict[str, Any]:
        """Get memory information and usage.

        Returns:
            Memory metrics including total, available, and usage percentage
        """
        if not PSUTIL_AVAILABLE:
            return {'available': False, 'error': 'psutil not installed'}

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'available': True,
            'virtual': {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'percent': mem.percent,
            },
            'swap': {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent,
            },
        }

    def get_disk_info(self) -> dict[str, Any]:
        """Get disk usage information.

        Returns:
            Disk metrics including total, used, and free space
        """
        if not PSUTIL_AVAILABLE:
            return {'available': False, 'error': 'psutil not installed'}

        partitions = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent,
                })
            except PermissionError:
                # Skip partitions we can't access
                continue

        return {
            'available': True,
            'partitions': partitions,
        }

    def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU information and usage.

        Returns:
            GPU metrics if available, otherwise error message
        """
        if not GPU_AVAILABLE:
            return {'available': False, 'error': 'GPUtil not installed'}

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {'available': False, 'error': 'No GPUs detected'}

            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory': {
                        'total': gpu.memoryTotal,
                        'used': gpu.memoryUsed,
                        'free': gpu.memoryFree,
                        'percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    },
                    'temperature': gpu.temperature,
                })

            return {
                'available': True,
                'count': len(gpus),
                'gpus': gpu_info,
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_system_info(self) -> dict[str, Any]:
        """Get general system information.

        Returns:
            System information including platform, architecture, and hostname
        """
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all system metrics.

        Returns:
            Complete system metrics including CPU, memory, disk, GPU, and system info
        """
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system': self.get_system_info(),
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'gpu': self.get_gpu_info(),
        }


# Global monitor instance
monitor = ResourceMonitor()
