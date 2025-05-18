import uuid
import boto3
from botocore.exceptions import ClientError


AWS_REGION        = "ap-southeast-1"        # ← vùng đang chạy Bedrock
AGENT_ID          = "S5AN8U5L8H"    # ← ID thật của Agent
AGENT_ALIAS_ID    = "FFP4TGX6GC"             # ← ID alias đã “Deploy”
PROMPT            = "Ngân hàng SHB có bao nhiêu khách hàng cá nhân?"
SESSION_ID        = str(uuid.uuid4())      # Giữ nguyên để tiếp tục hội thoại


def build_client(region: str = AWS_REGION):
    """
    Tạo client bedrock‑agent‑runtime.
    """
    return boto3.client("bedrock-agent-runtime", region_name=region,
                        aws_access_key_id="sample",
                        aws_secret_access_key="sample"
                        )


def invoke_agent(
    client,
    agent_id: str,
    alias_id: str,
    prompt: str,
    session_id: str,
    *,
    stream: bool = True,
    enable_trace: bool = True,
):
    """
    Gửi prompt tới Agent, nhận phản hồi,
    in trace (nếu có) và trả về chuỗi câu trả lời.
    """
    kwargs = {
        "agentId": agent_id,
        "agentAliasId": alias_id,
        "inputText": prompt,
        "sessionId": session_id,
        "enableTrace": enable_trace,
    }

    # Cấu hình streaming (tùy chọn)
    if stream:
        kwargs["streamingConfigurations"] = {
            "streamFinalResponse": False,          # False ⇒ nhận nhiều chunk
            "applyGuardrailInterval": 50           # mỗi 50 ký tự mới gọi Guardrail
        }

    response_stream = client.invoke_agent(**kwargs)  # sự kiện stream (EventStream)

    completion = ""
    for event in response_stream.get("completion"):
        if "chunk" in event:
            completion += event["chunk"]["bytes"].decode()


    return completion.strip()


if __name__ == "__main__":
    try:
        bedrock_client = build_client()
        answer = invoke_agent(
            bedrock_client,
            AGENT_ID,
            AGENT_ALIAS_ID,
            PROMPT,
            SESSION_ID,
            stream=True,
            enable_trace=True,
        )
        print("\n⭐ Phản hồi của Agent:\n", answer)

    except ClientError as err:
        print("Lỗi AWS:", err)
