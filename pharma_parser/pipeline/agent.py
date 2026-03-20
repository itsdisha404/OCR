"""
agent.py
────────
Stage 4: LangChain ReAct agent for error recovery.
Triggered only when overall_confidence == "LOW".
Uses OpenRouter (cloud) or Ollama (local) via .env config.
"""

import os
import re
import json

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm():
    """Create a LangChain ChatOpenAI instance pointing to OpenRouter or Ollama."""
    if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        return ChatOpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            model=os.getenv("OLLAMA_MODEL", "phi3:mini"),
            temperature=0.0,
            max_tokens=2000,
        )
    else:
        return ChatOpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
            temperature=0.0,
            max_tokens=2000,
        )


# ── Agent tools ───────────────────────────────────────────────────────────────

@tool
def lookup_hsn_code(product_name: str) -> str:
    """
    Given a pharma product name, returns the most likely HSN code.
    Use when hsn_code is empty or flagged as INVALID_HSN.
    """
    name_upper = product_name.upper()
    if any(k in name_upper for k in ["TABLET", "TAB", "CAPSULE", "CAP", "SYRUP", "INJECTION", "INJ"]):
        return "30049099"
    if any(k in name_upper for k in ["CREAM", "OINTMENT", "GEL", "LOTION", "SOAP"]):
        return "33049990"
    if any(k in name_upper for k in ["TOOTHPASTE", "MOUTHWASH"]):
        return "33061020"
    if any(k in name_upper for k in ["POWDER", "SUPPLEMENT", "NUTRITION", "PROTEIN"]):
        return "21069099"
    if any(k in name_upper for k in ["DROP", "EYE", "OPHTHALMIC"]):
        return "30049039"
    return "30049099"  # Default: Medicaments NES


@tool
def fix_gst_math(taxable_value: float, cgst_rate: float, sgst_rate: float) -> dict:
    """
    Recalculate correct GST amounts from taxable value and rates.
    Use when CGST_MISMATCH or SGST_MISMATCH flags are present.
    """
    cgst = round(taxable_value * cgst_rate / 100, 2)
    sgst = round(taxable_value * sgst_rate / 100, 2)
    return {
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "total_gst": round(cgst + sgst, 2),
        "total_amount": round(taxable_value + cgst + sgst, 2),
    }


@tool
def validate_gstin(gstin: str) -> str:
    """Validate a GSTIN number format. Returns 'VALID' or 'INVALID: reason'."""
    if not gstin:
        return "INVALID: empty"
    if not re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$", gstin):
        return f"INVALID: format mismatch ({len(gstin)} chars, expected 15)"
    return "VALID"


@tool
def correct_product_name(raw_name: str) -> str:
    """Correct OCR-mangled or misspelled pharma product names."""
    name = raw_name.upper().strip()
    fixes = {
        "TELSARTAN": "TELMISARTAN",
        "ATENOIOL": "ATENOLOL",
        "AMLODIPIN": "AMLODIPINE",
        "METFORRMIN": "METFORMIN",
        "PANTOPRAZOL": "PANTOPRAZOLE",
        "OMEPRAZOL": "OMEPRAZOLE",
        "LEVOCETRIZINE": "LEVOCETIRIZINE",
        "AZITHROMYCN": "AZITHROMYCIN",
        "CEFIXME": "CEFIXIME",
        "CIPROFLOXACN": "CIPROFLOXACIN",
    }
    for bad, good in fixes.items():
        if bad in name:
            name = name.replace(bad, good)
    return name.title()


@tool
def flag_for_human_review(invoice_no: str, reason: str) -> str:
    """
    Mark an invoice for human review when the agent cannot fix it.
    Use as LAST resort when all other tools have failed.
    """
    return f"FLAGGED:{invoice_no}:{reason}"


# ── Agent prompt ──────────────────────────────────────────────────────────────

AGENT_PROMPT = PromptTemplate.from_template("""
You are a pharma invoice data quality agent for an Indian pharmaceutical distributor.
You receive an invoice that failed automated validation. Fix all issues using tools.

IMPORTANT RULES:
- GST rates in Indian pharma: 5% (CGST 2.5% + SGST 2.5%) or 18% (CGST 9% + SGST 9%)
- HSN codes for medicines: 30049099 (tablets/capsules), 30049069 (topical)
- HSN codes for food supplements: 21069099
- Always fix GST math if taxable_value is correct but amounts are wrong
- If you cannot determine a value, use flag_for_human_review

INVOICE WITH ISSUES:
{invoice_json}

VALIDATION FLAGS:
{flags}

You have access to these tools:
{tools}

Tool names: {tool_names}

Use this format:
Thought: [what needs fixing]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output]
... (repeat as needed)
Thought: I have fixed all fixable issues
Final Answer: [the corrected invoice JSON]

{agent_scratchpad}
""")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_error_agent(invoice: dict) -> dict:
    """
    Take a LOW-confidence invoice, run the ReAct agent to fix it,
    return the corrected invoice dict.
    """
    tools = [
        lookup_hsn_code,
        fix_gst_math,
        validate_gstin,
        correct_product_name,
        flag_for_human_review,
    ]

    llm = get_llm()
    agent = create_react_agent(llm, tools, AGENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    # Collect all flags
    all_flags = (
        invoice.get("validation_summary", {}).get("header_flags", [])
        + [f for item in invoice.get("line_items", []) for f in item.get("flags", [])]
        + invoice.get("validation_summary", {}).get("summary_flags", [])
    )

    try:
        result = executor.invoke({
            "invoice_json": json.dumps(invoice, indent=2, default=str)[:4000],
            "flags": "\n".join(f"- {f}" for f in all_flags),
        })
        output = result.get("output", "")

        # Try to parse agent's JSON output
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            fixed = json.loads(json_match.group(0))
            fixed["agent_applied"] = True
            fixed["agent_steps"] = len(result.get("intermediate_steps", []))
            return fixed

    except Exception as e:
        invoice["agent_error"] = str(e)

    invoice.setdefault("validation_summary", {}).setdefault("header_flags", []).append("AGENT_FAILED")
    return invoice
