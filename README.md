# Transformer Training Monitor

This set of tools helps you monitor system resources during machine learning training, with special features for macOS and PyTorch MPS devices.

## Main Features

- System resource monitoring (CPU, memory, disk I/O, network)
- Enhanced PyTorch MPS GPU memory monitoring on Apple Silicon Macs
- Automatic detection of ML training processes
- Visual representation with ASCII bar charts and color-coded metrics
- Trend indicators to see resource usage patterns (↑, ↓, →)
- Customizable visualization options
- CSV logging for later analysis

## Tools Overview

1.  **`monitor_training.py`** - Main monitoring script. Interactively displays system and process-specific metrics.
2.  **`list_python_processes.py`** - Utility script used by `monitor_training.py` to identify relevant Python processes.
3.  **`test_monitoring.py`** - Script to test the monitoring functionality.
4.  **`test_process_detection.sh`** & **`verify_process_detection.py`** - Scripts for testing and verifying the process detection capabilities.

## Basic Usage

To start monitoring, run:

```bash
python3 monitor_training.py --auto-detect
```

This command will automatically detect and monitor Python processes that appear to be related to ML training.

## Configuration Options (`monitor_training.py`)

| Option                | Description                                         |
| --------------------- | --------------------------------------------------- |
| `--interval SECONDS`  | Update interval in seconds (default: 2).            |
| `--log FILEPATH`      | Log data to a specified CSV file.                   |
| `--processes COUNT`   | Maximum number of processes to show.                |
| `--save-config`       | Save current settings as default configuration.     |
| `--auto-detect`       | Auto-detect training processes.                     |
| `--enable-debug`      | Enable debug output for troubleshooting.            |
| `--watch-pid PID`     | Watch a specific process ID.                        |
| `--no-color`          | Disable colored output in the terminal.             |
| `--compact`           | Use a compact display format.                       |
| `--viz-width WIDTH`   | Width of ASCII visualization bars.                  |
| `--no-process-scan` | Disable Python process scanning (if watching a PID). |

## PyTorch MPS GPU Monitoring

The tool provides enhanced monitoring for PyTorch MPS devices on Apple Silicon Macs:

- Automatic detection of MPS device.
- Tracking of memory allocated by PyTorch on the MPS device.
- Estimation of total available MPS memory and usage percentage.
- Tensor counting and memory usage by tensors on the MPS device (experimental).
- Graceful error handling if MPS is not available or not initialized.

## Log File Analysis

When using the `--log` option, monitoring data is saved in CSV format. This data can be easily loaded into analysis tools like Pandas in Python for plotting or further investigation.

Example of analyzing logs:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the log data
df = pd.read_csv('your_log_file.csv')

# Separate system and process metrics if needed (check log structure)
# For example, if there's a 'type' column:
# system_metrics = df[df['type'] == 'system']
# process_metrics = df[df['type'] == 'process']

# Assuming direct columns for system CPU and GPU usage
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df['timestamp'], df['cpu_percent'], label='CPU Usage')
plt.title('CPU Usage Over Time')
plt.xlabel('Time')
plt.ylabel('CPU %')
plt.grid(True)
plt.legend()

# Adjust column name for GPU based on your log file (e.g., 'gpu_percent', 'mps_memory_allocated_gb')
if 'gpu_percent' in df.columns:
    plt.subplot(1, 2, 2)
    plt.plot(df['timestamp'], df['gpu_percent'], label='GPU Usage', color='orange')
    plt.title('GPU Usage Over Time')
    plt.xlabel('Time')
    plt.ylabel('GPU %')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
# plt.savefig('resource_usage.png') # Optionally save the plot
```

## Tips for Effective Monitoring

1.  **Use `--auto-detect`** for automatic process detection. The tool will attempt to identify Python processes running common ML frameworks or scripts.
2.  **Enable logging (`--log`)** for post-training analysis or if the terminal output is too verbose to watch live.
3.  **Use compact mode (`--compact`)** on smaller screens or when monitoring many processes.
4.  **Adjust visualization width (`--viz-width`)** to change the size of ASCII bar charts to fit your terminal.
5.  **Save your preferred settings (`--save-config`)** to avoid re-typing common options. The configuration is typically saved to `~/.monitor_training_config.json`.

## Troubleshooting

### Process Not Detected

If your training process is not automatically detected:
- Use `--enable-debug` to see process scanning details and understand why a process might be missed.
- Specify the Process ID (PID) directly with `--watch-pid YOUR_PID` if you know it.

### MPS Memory Issues

If you encounter issues with MPS memory reporting:
- Ensure PyTorch is correctly installed and configured with MPS support on your Apple Silicon Mac.
- Try running with `--enable-debug` for more detailed output from the MPS monitoring components.
- Check that your version of PyTorch and macOS are compatible and support the MPS backend features used by the monitor.

### Script Crashes or Errors

If the monitoring script crashes or shows errors:
- Ensure `psutil` is installed and up-to-date: `pip install --upgrade psutil`.
- Verify you have the necessary permissions to access process information.
- Try running with fewer features enabled (e.g., `--no-process-scan` if you provide a PID, or without MPS-specific features if on a different platform).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
