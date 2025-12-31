"""
QBM Orchestrator for Azure OpenAI Function Calling

Phase 6: Tool-first orchestration with function-calling.
Coordinates between Azure OpenAI and QBM tools with verification.

Flow:
1. Receive question from user
2. Send to Azure OpenAI with tool definitions
3. Execute tool calls and verify outputs
4. Return verified response or fail-closed error
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tools and verifier
from .tools import QBMTools, TOOL_DEFINITIONS
from .verifier import CitationVerifier, VerificationResult, fail_closed_gate

# Azure OpenAI imports
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure OpenAI SDK not available")


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_AR = """أنت عالم متخصص في تحليل السلوك القرآني وفق منهجية مصفوفة السلوك القرآني (QBM).

## القواعد الصارمة:
1. استخدم الأدوات المتاحة للحصول على البيانات - لا تخترع معلومات
2. كل ادعاء يجب أن يكون مدعوماً بآية قرآنية (verse_key)
3. استشهد بالتفاسير من المصادر السبعة المعتمدة فقط
4. لا تستخدم آيات الفاتحة أو أوائل البقرة كدليل أساسي
5. إذا لم تجد معلومات، قل ذلك بوضوح

## الأدوات المتاحة:
- resolve_entity: لتحويل المصطلح إلى معرف السلوك
- get_behavior_dossier: للحصول على ملف السلوك الكامل
- get_causal_paths: لإيجاد السلاسل السببية
- get_tafsir_comparison: لمقارنة التفاسير
- get_graph_metrics: لإحصائيات الرسم البياني
- get_verse_evidence: للحصول على نصوص الآيات

## المصادر المعتمدة:
ابن كثير، الطبري، القرطبي، السعدي، الجلالين، البغوي، الميسر"""

SYSTEM_PROMPT_EN = """You are a scholar specializing in Quranic behavioral analysis using the QBM (Quranic Behavioral Matrix) methodology.

## Strict Rules:
1. Use the available tools to get data - do not fabricate information
2. Every claim must be supported by a Quranic verse (verse_key)
3. Only cite tafsir from the 7 canonical sources
4. Do not use Fatiha or early Baqarah verses as primary evidence
5. If you cannot find information, say so clearly

## Available Tools:
- resolve_entity: Convert term to behavior ID
- get_behavior_dossier: Get complete behavior profile
- get_causal_paths: Find causal chains between behaviors
- get_tafsir_comparison: Compare tafsir sources
- get_graph_metrics: Get graph statistics
- get_verse_evidence: Get verse texts

## Canonical Sources:
Ibn Kathir, Tabari, Qurtubi, Sa'di, Jalalayn, Baghawi, Muyassar"""


# =============================================================================
# QBM ORCHESTRATOR
# =============================================================================

class QBMOrchestrator:
    """
    Orchestrates QBM queries through Azure OpenAI with function calling.
    
    Features:
    - Tool-first approach: LLM uses tools to get data
    - Verification gate: All outputs verified before returning
    - Fail-closed: Invalid responses blocked
    - Multi-turn: Supports iterative tool calls
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        language: str = "ar",
        data_dir: Optional[Path] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            api_key: Azure OpenAI API key (or from env)
            endpoint: Azure OpenAI endpoint (or from env)
            deployment: Deployment name (or from env)
            api_version: API version
            language: Response language (ar/en)
            data_dir: Data directory for tools
        """
        self.language = language
        self.system_prompt = SYSTEM_PROMPT_AR if language == "ar" else SYSTEM_PROMPT_EN
        
        # Initialize tools
        self.tools = QBMTools(data_dir=data_dir)
        self.verifier = CitationVerifier(data_dir=data_dir)
        
        # Initialize Azure OpenAI client
        self.client = None
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        if AZURE_AVAILABLE:
            try:
                self.client = AzureOpenAI(
                    api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                    azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                logger.info(f"Azure OpenAI client initialized (deployment: {self.deployment})")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI: {e}")
                self.client = None
        
        # Conversation history for multi-turn
        self.messages: List[Dict[str, Any]] = []
        self.max_tool_calls = 10  # Prevent infinite loops
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = []
    
    def query(
        self,
        question: str,
        verify: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Process a question through the orchestrator.
        
        Args:
            question: User question
            verify: Whether to verify the response
            max_tokens: Maximum response tokens
            temperature: Response temperature
            
        Returns:
            Response with answer, tool_calls, and verification status
        """
        if not self.client:
            return {
                "success": False,
                "error": "Azure OpenAI client not initialized",
                "message": "Check AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables"
            }
        
        # Initialize conversation
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        tool_calls_made = []
        tool_outputs = []
        
        # Iterative tool calling loop
        for iteration in range(self.max_tool_calls):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=self.messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                message = response.choices[0].message
                
                # Check if model wants to call tools
                if message.tool_calls:
                    # Add assistant message with tool calls
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })
                    
                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                        
                        # Execute tool
                        tool_result = self.tools.execute_tool(tool_name, arguments)
                        
                        # Verify tool output if requested
                        if verify:
                            verification = self.verifier.verify_tool_output(tool_name, tool_result)
                            if not verification.valid:
                                logger.warning(f"Tool output verification failed: {verification.violations}")
                                tool_result["_verification_warnings"] = verification.violations
                        
                        tool_calls_made.append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "success": tool_result.get("success", False)
                        })
                        tool_outputs.append(tool_result)
                        
                        # Add tool result to messages
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        })
                else:
                    # No more tool calls, model has final answer
                    final_answer = message.content
                    
                    # Build response
                    result = {
                        "success": True,
                        "answer": final_answer,
                        "tool_calls": tool_calls_made,
                        "tool_outputs": tool_outputs,
                        "iterations": iteration + 1,
                        "provenance": {
                            "tools_used": [tc["tool"] for tc in tool_calls_made],
                            "model": self.deployment,
                            "language": self.language
                        }
                    }
                    
                    # Final verification
                    if verify:
                        verification = self.verifier.verify_response(result)
                        result["verification"] = verification.to_dict()
                        
                        if not verification.valid:
                            # Fail-closed: return error instead of unverified response
                            return fail_closed_gate(result, self.verifier)
                    
                    return result
                    
            except Exception as e:
                logger.error(f"Error in orchestrator iteration {iteration}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "iteration": iteration,
                    "tool_calls": tool_calls_made
                }
        
        # Max iterations reached
        return {
            "success": False,
            "error": "Max tool call iterations reached",
            "tool_calls": tool_calls_made,
            "tool_outputs": tool_outputs
        }
    
    def query_with_tools_only(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        verify: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a single tool directly without LLM.
        
        Useful for testing and direct data access.
        
        Args:
            tool_name: Tool to execute
            arguments: Tool arguments
            verify: Whether to verify output
            
        Returns:
            Tool output with optional verification
        """
        result = self.tools.execute_tool(tool_name, arguments)
        
        if verify:
            verification = self.verifier.verify_tool_output(tool_name, result)
            result["verification"] = verification.to_dict()
            
            if not verification.valid:
                return fail_closed_gate(result, self.verifier)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "azure_client": "ready" if self.client else "not configured",
            "deployment": self.deployment,
            "language": self.language,
            "tools_available": list(self.tools.execute_tool.__code__.co_consts),
            "verifier": "ready",
            "max_tool_calls": self.max_tool_calls
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_orchestrator(language: str = "ar") -> QBMOrchestrator:
    """
    Create an orchestrator with default configuration.
    
    Reads configuration from environment variables:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_DEPLOYMENT
    
    Args:
        language: Response language (ar/en)
        
    Returns:
        Configured QBMOrchestrator
    """
    return QBMOrchestrator(language=language)


def quick_query(question: str, language: str = "ar") -> Dict[str, Any]:
    """
    Quick query function for simple use cases.
    
    Args:
        question: Question to ask
        language: Response language
        
    Returns:
        Query result
    """
    orchestrator = create_orchestrator(language=language)
    return orchestrator.query(question)
