# demo_client.py — Minimal Python MCP client (works with mcp>=1.16)

import sys
import anyio

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

BRAND = sys.argv[1] if len(sys.argv) > 1 else "Apple"

async def main() -> None:
    # Launch server.py with the CURRENT interpreter
    params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"]
    )

    async with stdio_client(params) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()

        # List tools
        tools = await session.list_tools()
        print("\nTools available:")
        for t in tools.tools:
            print(" -", t.name)

        # Count mentions
        print(f"\nCalling count_brand_mentions for '{BRAND}'...")
        result = await session.call_tool("count_brand_mentions", {"brand": BRAND})
        if getattr(result, "content", None):
            for c in result.content:
                text = getattr(c, "text", None)
                if text:
                    print(text)
        else:
            print(result)

        # Sample snippets
        print("\nSample snippets:")
        samples = await session.call_tool("sample_reviews_with_brand", {"brand": BRAND, "limit": 3})
        if getattr(samples, "content", None):
            for c in samples.content:
                text = getattr(c, "text", None)
                if text:
                    print(text)
        else:
            print(samples)

        # Optional LLM explnation
        print("\nLLM explanation (optional):")
        try:
            explain = await session.call_tool("explain_result_with_llm", {"brand": BRAND})
            if getattr(explain, "content", None):
                for c in explain.content:
                    text = getattr(c, "text", None)
                    if text:
                        print(text)
            else:
                print(explain)
        except Exception as e:
            print("Skipped LLM explanation:", e)

if __name__ == "__main__":
    anyio.run(main)
