import json
import os
from typing import Any, Dict
import re
import arxiv
from dotenv import load_dotenv
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
load_dotenv()

def download_paper(max_results=25):
    """
    下载 arXiv 上最新的 LLM 相关论文。
    基于arxiv官方api实现
    目前arxiv不支持根据日期精准搜索，不过据我观察llm based agent领域一天十几篇，因此每次下载25篇并跳过之前重复过的
    
    参数:
      max_results (int): 下载的论文最大数量，默认为 25。
    """
    download_dir = r'TFSAgent\\paper'
    ids_dir = r'TFSAgent\\ids'
    # 确保 ids 目录存在
    if not os.path.exists(ids_dir):
        os.makedirs(ids_dir)
    ids_file = os.path.join(ids_dir, 'downloaded_ids.txt')


    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    
    downloaded_ids = set()
    if os.path.exists(ids_file):
        with open(ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                downloaded_ids.add(line.strip())

    client = arxiv.Client()
    search = arxiv.Search(
        query="all:llm AND all:agent",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for r in client.results(search):

        paper_id = r.get_short_id() if hasattr(r, "get_short_id") else r.entry_id.split('/')[-1]
        if paper_id not in downloaded_ids:
            
            logger.debug(f"[TFS😋]Downloading new paper: {paper_id}")
            r.download_pdf(dirpath=download_dir)
            downloaded_ids.add(paper_id)
        else:
           
            logger.debug(f"[TFS😋]Skipped {paper_id}, already processsed.")


    with open(ids_file, 'w', encoding='utf-8') as f:
        for pid in downloaded_ids:
            f.write(pid + "\n")


def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # For any other type, convert to string
        return str(obj)


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace('\\"', "'")
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")

import pdfplumber
import json

import pdfplumber

def group_words_to_lines(words, threshold=3):
    """
    将 pdfplumber 提取的 word 列表按行聚合，
    threshold 为相同行之间允许的 y 坐标差异。
    返回列表，每一项为一行的文本及其大致 y 坐标。
    """
    lines = []
    # 先按照 top 坐标排序
    words = sorted(words, key=lambda w: w["top"])
    
    for word in words:
        placed = False
        for line in lines:
            # 如果 word 与该行的 top 坐标差距小于 threshold，则归到同一行
            if abs(word["top"] - line["top"]) < threshold:
                line["words"].append(word)
                # 更新行的 top 为平均值（可选）
                line["top"] = (line["top"] * (len(line["words"]) - 1) + word["top"]) / len(line["words"])
                placed = True
                break
        if not placed:
            # 新建一行
            lines.append({"top": word["top"], "words": [word]})
    # 对每行内的单词按 x0 坐标排序，并生成文本
    result = []
    for line in lines:
        line_words = sorted(line["words"], key=lambda w: w["x0"])
        text_line = " ".join(w["text"] for w in line_words)
        result.append({"top": line["top"], "type": "text", "content": text_line})
    return result

def pdf_to_text_with_structure(pdf_path: str) -> None:
    """
    从 PDF 中提取纯文本和结构化表格数据，按照在页面中的垂直顺序合并，
    最后输出到一个 TXT 文件中。
    
    表格会以 [Table] 标记输出，每行的各单元格之间以制表符分隔。
    """
    output_lines = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 添加页眉信息
            output_lines.append(f"=== Page {page.page_number} ===")
            
            # 提取页面所有单词，并聚合成行
            words = page.extract_words()
            text_blocks = group_words_to_lines(words)
            
            # 查找页面中的表格，pdfplumber 的 find_tables() 返回 Table 对象列表，
            # 每个 Table 对象有 bbox 属性和 extract() 方法提取二维列表数据
            table_blocks = []
            tables = page.find_tables()
            for table in tables:
                bbox = table.bbox  # (x0, top, x1, bottom)
                # 使用表格所在区域的 top 作为排序依据
                table_data = table.extract()
                table_blocks.append({"top": bbox[1], "type": "table", "content": table_data})
            
            # 合并文本块和表格块，并按 top 坐标排序
            blocks = text_blocks + table_blocks
            blocks = sorted(blocks, key=lambda b: b["top"])
            
            # 遍历合并后的块，构造输出
            for block in blocks:
                if block["type"] == "text":
                    output_lines.append(block["content"])
                elif block["type"] == "table":
                    output_lines.append("[Table]:")
                    # 将表格每一行按制表符连接
                    for row in block["content"]:
                        output_lines.append("\t".join(cell if cell is not None else "" for cell in row))
            output_lines.append("")  # 空行分隔页面内容
    return output_lines



import requests
import json
# notion_token = os.environ.get("NOTION_TOKEN")
# database_id = os.environ.get("NOTION_DATABASE_ID")
def insert_to_notion(data: dict) -> None:
    """
    将包含论文信息的 JSON 数据插入 Notion 数据库中。
    
    参数：
      data: 包含论文信息的字典，例如：
            {
                "upload_time": "2023-07-28T10:00:00+08:00",
                "paper_title": "某论文标题",
                "abstract": "论文的摘要内容……",
                "research_deficiencies": "当前研究存在的问题……"
            }
      notion_token: 你的 Notion 集成的 API Token（Bearer Token）。
      database_id: Notion 数据库的 ID。
    """
    url = "https://api.notion.com/v1/pages"
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Id": {
                "title": [{ "type": "text", "text": { "content": "Tomatoes" } }]
            },
            "Upload Time": {
                "date": {"start": data.get("upload_time")}
            },

            "Paper Title": {
                "rich_text": [
                    {
                        "text": {"content": data.get("paper_title", "")}
                    }
                ]
            },
            "Abstract": {
                "rich_text": [
                    {
                        "text": {"content": data.get("abstract", "")}
                    }
                ]
            },
            "Research Deficiencies": {
                "rich_text": [
                    {
                        "text": {"content": data.get("research_deficiencies", "")}
                    }
                ]
            },
            "Methodology": {
                "rich_text": [
                    {
                        "text": {"content": data.get("Methodology", "")}
                    }
                ]
            },
            "Experiments": {
                "rich_text": [
                    {
                        "text": {"content": data.get("Experiments", "")}
                    }
                ]
            },
            "Conclusion": {
                "rich_text": [
                    {
                        "text": {"content": data.get("Conclusion", "")}
                    }
                ]
            }
            
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info(f"[TFS😋]: Notion database updated successfully")
    else:
        logger.error(f"[TFS😨]: Notion database updated failed: {response.status_code}")
        logger.error(f"[TFS😨]: err: {response.text}")




