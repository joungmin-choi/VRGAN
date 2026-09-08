import os
import shutil
import subprocess

if os.path.exists("./results"):
    shutil.rmtree("./results")


def test_VRGAN_runs():
    # Run the CLI end-to-end on the bundled example dataset with a small
    # epoch budget, just to confirm the pipeline executes cleanly.
    result = subprocess.run(
        ["python3", "VRGAN.py", "./example", "./results", "20", "sd", "20", "10"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Process failed: {result.stderr}"
    assert os.path.exists("./results"), "Results directory was not created"
    assert os.path.exists("./results/anomaly_score_test.csv")
    assert os.path.exists("./results/vrnn_reconstruction_test.csv")
    assert os.path.exists("./results/gan_reconstruction_test.csv")
    assert os.path.exists("./results/threshold.txt")
