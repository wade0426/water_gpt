import requests
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate as CorePromptTemplate # For agent's internal prompt
from langchain.llms.base import LLM
import json
import os
import shutil # For rename_file

# --- 設定 ---
API_URL = "http://4090p8000.huannago.com/v1/chat/completions" # 請確認此 URL 是否對外可訪問或您有權限
HEADERS = {"Content-Type": "application/json"}
MODEL   = "gemma3-14b-it" # 請確認此模型是否適合 Agent 的推理需求

# --- 自定義 LLM for Agent ---
class AgentLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom_agent_llm"

    def _call(self, prompt: str, stop=None) -> str:
        # Agent 的 prompt 可能會包含工具描述、思考過程等。
        # 這裡的 system message 可以比較通用，或者由 LangChain 的 Agent prompt 提供。
        # 為了簡單起見，我們讓 LangChain 的 agent prompt format 來處理大部分引導
        system_message_content = """You are a helpful AI assistant that strictly follows the ReAct (Reasoning and Acting) framework."""
        payload = {
            "model":    MODEL,
            "messages": [
                # LangChain 的 react agent 會構建一個包含 "Thought:", "Action:", "Action Input:", "Observation:" 的 prompt
                # 我們直接將這個 prompt 作為 user message 傳遞
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.0 # 建議 Agent 使用較低的 temperature 以獲得更可預測的工具調用
        }
        # print(f"\n---- LLM Input ----\n{prompt}\n---- END LLM Input ----\n") # 用於調試
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60) # 增加 timeout
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            # print(f"\n---- LLM Output ----\n{response_text}\n---- END LLM Output ----\n") # 用於調試
            return response_text
        except requests.exceptions.RequestException as e:
            return f"API Error: {e}"
        except (KeyError, IndexError) as e:
            return f"API Response parsing error: {e}, Response: {data}"


    @property
    def identifying_params(self) -> dict:
        return {"model": MODEL, "api_url": API_URL}

# --- 工具函數定義 ---
def _read_file(file_path: str) -> str:
    """
    讀取指定路徑檔案的內容。
    輸入：檔案的完整路徑。
    """
    try:
        # 為了安全，限制只能讀取當前目錄下的檔案
        if ".." in file_path or os.path.isabs(file_path):
             return "錯誤：處於安全考量，只能讀取當前工作目錄下的相對路徑檔案。"
        
        # 確保檔案在當前目錄或子目錄
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(os.getcwd()):
            return "錯誤：處於安全考量，只能讀取當前工作目錄或其子目錄下的檔案。"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"檔案 '{file_path}' 的內容是:\n{content}"
    except FileNotFoundError:
        return f"錯誤：找不到檔案 '{file_path}'。"
    except Exception as e:
        return f"讀取檔案 '{file_path}' 時發生錯誤: {e}"

def _list_files(directory_path: str = ".") -> str:
    """
    列出指定目錄中的檔案和資料夾。
    輸入：要列出內容的目錄路徑 (可選，預設為當前目錄)。
    """
    try:
        # 為了安全，限制只能列出當前目錄下的內容
        if ".." in directory_path or os.path.isabs(directory_path):
             return "錯誤：處於安全考量，只能列出當前工作目錄下的相對路徑目錄。"
        
        abs_path = os.path.abspath(directory_path)
        if not abs_path.startswith(os.getcwd()):
            return "錯誤：處於安全考量，只能列出當前工作目錄或其子目錄下的內容。"

        if not os.path.isdir(directory_path):
            return f"錯誤：'{directory_path}' 不是一個有效的目錄。"
        
        items = os.listdir(directory_path)
        if not items:
            return f"目錄 '{directory_path}' 是空的。"
        return f"目錄 '{directory_path}' 中的內容:\n" + "\n".join(items)
    except Exception as e:
        return f"列出目錄 '{directory_path}' 時發生錯誤: {e}"

def _rename_file(path_args: str) -> str:
    """
    重命名檔案或資料夾。
    輸入：一個包含舊路徑和新路徑的字串，以逗號分隔，例如 'old_name.txt,new_name.txt'。
    """
    try:
        parts = path_args.split(',')
        if len(parts) != 2:
            return "錯誤：重命名需要提供兩個參數：舊路徑和新路徑，並以逗號分隔。"
        
        old_path = parts[0].strip()
        new_path = parts[1].strip()

        # 安全性檢查
        for path_item in [old_path, new_path]:
            if ".." in path_item or os.path.isabs(path_item):
                return "錯誤：處於安全考量，路徑不能包含 '..' 或為絕對路徑。"
            abs_path_item = os.path.abspath(path_item)
            if not abs_path_item.startswith(os.getcwd()):
                 return "錯誤：處於安全考量，檔案操作限制在當前工作目錄或其子目錄下。"


        if not os.path.exists(old_path):
            return f"錯誤：找不到來源檔案或目錄 '{old_path}'。"
        if os.path.exists(new_path):
            return f"錯誤：目標路徑 '{new_path}' 已經存在。"

        shutil.move(old_path, new_path) # shutil.move 可以重命名檔案和資料夾
        return f"成功將 '{old_path}' 重命名為 '{new_path}'。"
    except Exception as e:
        return f"重命名時發生錯誤: {e}"

# --- 創建工具列表 ---
tools = [
    Tool(
        name="read_file",
        func=_read_file,
        description="""當你需要讀取一個檔案的內容時使用此工具。
        輸入應該是一個字串，代表檔案的相對路徑。例如：'my_document.txt' 或 'subdir/another_file.log'。
        只能操作當前工作目錄或其子目錄下的檔案。"""
    ),
    Tool(
        name="list_files",
        func=_list_files,
        description="""當你需要列出一個目錄中的所有檔案和子目錄時使用此工具。
        輸入可以是一個字串，代表目錄的相對路徑，例如 'my_folder' 或 '.' 代表當前目錄。
        如果沒有提供輸入，則預設列出當前工作目錄的內容。
        只能操作當前工作目錄或其子目錄。"""
    ),
    Tool(
        name="rename_file",
        func=_rename_file, # 注意這裡 func 的參數是一個字串
        description="""當你需要重命名一個檔案或資料夾時使用此工具。
        輸入應該是一個字串，包含舊的相對路徑和新的相對路徑，並用逗號分隔。
        例如：'old_filename.txt,new_filename.txt' 或 'old_folder,new_folder'。
        只能操作當前工作目錄或其子目錄下的檔案/資料夾。"""
    ),
]

# --- 初始化 Agent ---
# Agent 需要一個 LLM
llm = AgentLLM()

# Agent 需要一個 prompt 模板，LangChain 提供了默認的 ReAct prompt
# 我們需要從 langchain hub 拉取一個兼容的 prompt
# 或者手動定義一個，但 create_react_agent 會幫我們處理
# This is the prompt that will be used to instruct the LLM
# It is a template that will be filled with the tools and the input
# For ReAct agent, the prompt needs to include:
# - tools: a list of tools available
# - tool_names: a list of tool names
# - input: the user input
# - agent_scratchpad: the previous thought/action/observation steps
# LangChain's `create_react_agent` handles this logic.
# You can pull a default prompt from the hub:
# from langchain import hub
# prompt = hub.pull("hwchase17/react")

# 這裡我們使用 create_react_agent 來簡化 prompt 的創建
# 它會使用一個標準的 ReAct prompt 模板
try:
    from langchain import hub
    prompt_template = hub.pull("hwchase17/react")
except Exception as e:
    print(f"無法從 Langchain Hub 拉取 prompt: {e}")
    print("將使用手動定義的基礎 ReAct prompt。")
    # Fallback if hub is not accessible or has issues
    # This is a simplified version. A more robust one would handle {{}} escaping.
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    prompt_template = CorePromptTemplate.from_template(template)


agent = create_react_agent(llm, tools, prompt_template)

# AgentExecutor 負責運行 Agent 的思考-行動循環
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # 設為 True 可以看到 Agent 的思考過程
    handle_parsing_errors=True, # 處理 LLM 輸出格式錯誤的情況
    max_iterations=5 # 避免無限循環
)

# --- 測試 Agent ---
if __name__ == "__main__":
    # 為了測試，創建一些虛擬檔案和目錄
    if not os.path.exists("test_dir"):
        os.makedirs("test_dir")
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("這是 sample.txt 的內容。\nHello from LangChain!")
    with open("test_dir/another.txt", "w", encoding="utf-8") as f:
        f.write("這是 test_dir/another.txt 的內容。")
    if os.path.exists("old_name.txt"):
        os.remove("old_name.txt")
    if os.path.exists("new_name.txt"):
        os.remove("new_name.txt")
    with open("old_name.txt", "w", encoding="utf-8") as f:
        f.write("This file will be renamed.")

    print("\n--- 測試開始 ---")

    prompts_to_test = [
        "你好嗎？", # 閒聊，不應該調用工具
        "幫我列出當前目錄有什麼檔案",
        "讀取 sample.txt 這個檔案的內容。",
        "test_dir 目錄裡面有什麼？",
        "我想把 old_name.txt 改名成 new_name.txt",
        "讀一下 new_name.txt 的內容。", # 測試重命名後是否能讀取
        "讀取一個不存在的檔案叫做 ghost.txt",
        "把 sample.txt 重命名為 test_dir/another.txt", # 應該會失敗，因為目標已存在
        "列出 ../../some_other_dir 的檔案", # 應該會因為安全限制而失敗
        "讀取 /etc/passwd", # 應該會因為安全限制而失敗
    ]

    for user_prompt in prompts_to_test:
        print(f"\n\n>>>> 使用者提問: {user_prompt}")
        try:
            response = agent_executor.invoke({"input": user_prompt})
            print(f"<<<< Agent 回覆: {response['output']}")
        except Exception as e:
            print(f"處理 '{user_prompt}' 時發生錯誤: {e}")

    # 清理測試檔案
    print("\n--- 清理測試檔案 ---")
    if os.path.exists("sample.txt"): os.remove("sample.txt")
    if os.path.exists("new_name.txt"): os.remove("new_name.txt") # old_name.txt 已被重命名或刪除
    if os.path.exists("old_name.txt"): os.remove("old_name.txt") # 以防重命名失敗
    if os.path.exists("test_dir/another.txt"): os.remove("test_dir/another.txt")
    if os.path.exists("test_dir"): shutil.rmtree("test_dir")
    print("清理完成。")