import os
import time
import uuid
import json
import datetime
import logging
import argparse

from llm_api import query_llm
from utils import (
    download_paper,
    pdf_to_text_with_structure,
    insert_to_notion,
    parse_json_blob
)

# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 从 prompt 文件中读取系统提示
sys_prompt_file = "TFSAgent/prompt.txt"
if not os.path.exists(sys_prompt_file):
    logger.error("未找到系统提示文件: %s", sys_prompt_file)
    exit(1)
with open(sys_prompt_file, "r", encoding="utf-8") as f:
    sys_prompt = f.read().strip()


def call_llm(sys_prompt: str, user_prompt: str) -> str:
    response = query_llm(sys_prompt, user_prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agent：从 arXiv 下载论文，调用 LLM 解析，并上传至 Notion。"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=25,
        help="下载的论文最大数量，默认为 30"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=True,
        help="如果设置，则处理完后删除下载的 PDF 文件"
    )
    parser.add_argument(
        "--wait_time",
        type=int,
        default=20,
        help="下载后等待的秒数（默认 20 秒）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("[TFS😋]: Downloading papers from arXiv (max_results=%d)...", args.max_results)
    download_paper(max_results=args.max_results)
    logger.info("[TFS😋]: Papers downloading, waiting for %d seconds.", args.wait_time)
    time.sleep(args.wait_time)

    download_dir = r'TFSAgent\paper'
    pdf_files = [f for f in os.listdir(download_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.info("[TFS😨]: No PDF files found in the download directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(download_dir, pdf_file)
        logger.info("[TFS😋]: Processing PDF file: %s", pdf_file)

        # 提取 PDF 文本（保留页面结构）
        extracted_lines = pdf_to_text_with_structure(pdf_path)
        user_prompt = "\n".join(extracted_lines)

        # 调用 llm 接口，传入系统提示和用户提示
        llm_response = call_llm(sys_prompt, user_prompt)

        try:
            result_data = parse_json_blob(llm_response)
        except Exception as e:
            logger.error("[TFS😨]: Error parsing LLM response for %s: %s", pdf_file, e)
            continue

        # 系统自动设置上传时间，并生成唯一 id
        result_data["upload_time"] = datetime.datetime.now().isoformat()
        # 使用两个不同字段，分别可作为 Notion 页面唯一标识或其他用途
        result_data["id"] = str(uuid.uuid4())
        result_data["Id"] = str(uuid.uuid4())

        # 可将源 PDF 文件名加入数据包，方便追踪
        result_data["source_pdf"] = pdf_file

        # 上传数据至 Notion
        insert_to_notion(result_data)
        logger.info("[TFS😋]: Uploaded %s to Notion.", pdf_file)

    if args.delete:
        logger.info("[TFS😋]: Finished processing, deleting downloaded PDF files...")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(download_dir, pdf_file)
            os.remove(pdf_path)
        logger.info("[TFS😋]: PDF files deleted, agent task completed.")
    else:
        logger.info("[TFS😋]: Agent task completed, PDF files retained.")


if __name__ == "__main__":
    main()
