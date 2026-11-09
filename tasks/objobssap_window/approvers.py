from inspect_ai.approval import Approval, approver

@approver
def rename_nudge():
    async def approve(message, call, view, history):
        if call.function == "web_search_raw" and "objvissap" in str(call.arguments).lower():
            return Approval(decision="reject",
                            explanation="ObjVisSAP was renamed in 2024.")
        return Approval(decision="approve")
    return approve