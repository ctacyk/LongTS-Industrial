"""
将 LongTS-Industrial 基准数据集上传到 HuggingFace。

用法：
    set HF_TOKEN=hf_xxx                          # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
    python upload_benchmark_to_hf.py <用户名>/<数据集名>

例如：
    python upload_benchmark_to_hf.py your-name/LongTS-Industrial

说明：
    - Token 通过环境变量 HF_TOKEN 读取，不在代码中硬编码。
    - 默认上传 Data/benchmark_tb 整个目录（含 README 数据集卡片）。
    - 体量约 2.4GB，首次上传请确保网络稳定；huggingface_hub 支持断点续传。
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DATA_DIR = Path(__file__).parent / "Data" / "benchmark_tb"


def main():
    if len(sys.argv) < 2:
        print("用法: python upload_benchmark_to_hf.py <用户名>/<数据集名>")
        sys.exit(1)

    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("错误: 请先设置环境变量 HF_TOKEN")
        sys.exit(1)

    if not DATA_DIR.exists():
        print(f"错误: 找不到数据目录 {DATA_DIR}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"创建/确认数据集仓库: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"开始上传 {DATA_DIR} ...")
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload LongTS-Industrial benchmark",
    )
    print(f"完成: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
