# 🛡️ Guardrail Libraries Comparison

**Evaluation of enterprise-grade guardrail libraries vs. custom implementation**

---

## 📋 Executive Summary

Instead of building guardrails manually, you can use proven libraries that provide:
- ✅ Production-tested security patterns
- ✅ Regular updates for new attack vectors
- ✅ Community support and documentation
- ✅ Compliance certifications
- ✅ Faster time-to-market

**Recommendation**: **Hybrid approach** - Use libraries for input/output validation, keep custom logic for domain-specific policies.

---

## 🏆 Top Guardrail Libraries

### 1. **Guardrails AI** ⭐ RECOMMENDED

**Website**: https://www.guardrailsai.com  
**GitHub**: https://github.com/guardrails-ai/guardrails  
**License**: Apache 2.0

#### Features:
- ✅ **Structured output validation** (Pydantic integration)
- ✅ **60+ validators** (PII, toxicity, SQL injection, prompt injection)
- ✅ **Custom validators** (write your own in Python)
- ✅ **Retry logic** with fallbacks
- ✅ **RAG hallucination detection**
- ✅ **Multi-LLM support** (OpenAI, Anthropic, Cohere, local models)
- ✅ **Streaming support**
- ✅ **LangChain integration**

#### Code Example:
```python
from guardrails import Guard
from guardrails.hub import PII, ToxicLanguage, PromptInjection

# Create guard with multiple validators
guard = Guard().use_many(
    PII(pii_entities=["EMAIL", "PHONE", "SSN"]),
    ToxicLanguage(threshold=0.8),
    PromptInjection(threshold=0.9)
)

# Validate input
validated_output = guard.validate(
    llm_output="My email is john@example.com",
    metadata={"session_id": "123"}
)

if validated_output.validation_passed:
    print("Safe!")
else:
    print(f"Violations: {validated_output.error}")
```

#### Integration with Your Project:
```python
# src/guardrails/guardrails_ai_wrapper.py
from guardrails import Guard
from guardrails.hub import (
    PII, ToxicLanguage, PromptInjection, 
    SQLInjection, CodeSafety, SecretsDetection
)

class GuardrailsAIIntegration:
    def __init__(self):
        # Input guard
        self.input_guard = Guard().use_many(
            PII(pii_entities=["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]),
            PromptInjection(threshold=0.9),
            ToxicLanguage(threshold=0.8),
            SecretsDetection(patterns=["API_KEY", "TOKEN"])
        )
        
        # Output guard
        self.output_guard = Guard().use_many(
            CodeSafety(dangerous_patterns=["eval", "exec", "os.system"]),
            SQLInjection(),
            PII(pii_entities=["EMAIL", "PHONE", "SSN"])
        )
    
    def check_input(self, text: str) -> dict:
        result = self.input_guard.validate(text)
        return {
            "safe": result.validation_passed,
            "violations": result.error,
            "sanitized": result.validated_output
        }
    
    def check_output(self, code: str) -> dict:
        result = self.output_guard.validate(code)
        return {
            "safe": result.validation_passed,
            "issues": result.error,
            "fixed_code": result.validated_output
        }
```

#### Pros:
- ✅ Most mature library (3+ years in production)
- ✅ Largest validator hub (60+ validators)
- ✅ Active community (5.7K+ GitHub stars)
- ✅ LangChain/LlamaIndex integration
- ✅ Excellent documentation

#### Cons:
- ❌ No built-in HITL workflows
- ❌ No budget tracking
- ❌ Limited audit logging

#### Pricing:
- **Open Source**: Free
- **Guardrails Hub**: $99-$999/month for premium validators
- **Enterprise**: Custom pricing

---

### 2. **NeMo Guardrails** (NVIDIA)

**Website**: https://github.com/NVIDIA/NeMo-Guardrails  
**License**: Apache 2.0

#### Features:
- ✅ **Programmable guardrails** using Colang DSL
- ✅ **Fact-checking** against knowledge base
- ✅ **Jailbreak detection**
- ✅ **Hallucination detection**
- ✅ **Topical rails** (keep conversation on-topic)
- ✅ **Multi-turn conversation safety**
- ✅ **NVIDIA optimized** (fast on GPUs)

#### Code Example:
```python
from nemoguardrails import RailsConfig, LLMRails

# Define guardrails in Colang
config = RailsConfig.from_content(
    colang_content="""
    define user ask about personal info
      "What is your email?"
      "Give me your phone number"
    
    define bot refuse personal info
      "I cannot share personal information."
    
    define flow personal info guard
      user ask about personal info
      bot refuse personal info
      stop
    """,
    yaml_content="""
    models:
      - type: main
        engine: openai
        model: gpt-4
    
    rails:
      input:
        flows:
          - personal info guard
      output:
        flows:
          - check hallucination
    """
)

# Create rails
rails = LLMRails(config)

# Use with LLM
response = rails.generate(
    messages=[{"role": "user", "content": "What's your email?"}]
)
```

#### Integration:
```python
# src/guardrails/nemo_integration.py
from nemoguardrails import RailsConfig, LLMRails

class NeMoGuardrailsIntegration:
    def __init__(self):
        self.config = RailsConfig.from_path("./config/nemo_guardrails/")
        self.rails = LLMRails(self.config)
    
    def generate_with_guardrails(self, prompt: str) -> str:
        response = self.rails.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        return response["content"]
```

#### Pros:
- ✅ Declarative configuration (easier to maintain)
- ✅ Conversation-level guardrails (multi-turn)
- ✅ NVIDIA backing (enterprise support)
- ✅ Fast on GPUs

#### Cons:
- ❌ Steeper learning curve (Colang DSL)
- ❌ Less validator variety than Guardrails AI
- ❌ Newer library (less mature)

#### Pricing:
- **Open Source**: Free

---

### 3. **LLaMA Guard 2** (Meta)

**Website**: https://huggingface.co/meta-llama/LlamaGuard-2-8b  
**License**: Llama 2 Community License

#### Features:
- ✅ **LLM-based safety classifier** (8B parameter model)
- ✅ **12 safety categories** (violence, hate, sexual content, etc.)
- ✅ **Input + output moderation**
- ✅ **Fast inference** (optimized for real-time)
- ✅ **No API calls** (runs locally)
- ✅ **Multilingual support**

#### Code Example:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA Guard 2
model_id = "meta-llama/LlamaGuard-2-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def check_safety(text: str, role: str = "user") -> dict:
    """Check if text is safe."""
    prompt = f"<|{role}|> {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Parse response: "safe" or "unsafe\nS1,S3" (category codes)
    if response.strip().lower() == "safe":
        return {"safe": True, "categories": []}
    else:
        categories = response.split("\n")[1].split(",")
        return {"safe": False, "categories": categories}

# Usage
result = check_safety("How do I hack into a database?")
print(result)  # {'safe': False, 'categories': ['S9']}  # S9 = cybersecurity
```

#### Integration:
```python
# src/guardrails/llama_guard_integration.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaGuardIntegration:
    def __init__(self):
        model_id = "meta-llama/LlamaGuard-2-8b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def check_input(self, text: str) -> dict:
        return self._check_safety(text, "user")
    
    def check_output(self, text: str) -> dict:
        return self._check_safety(text, "assistant")
    
    def _check_safety(self, text: str, role: str) -> dict:
        prompt = f"<|{role}|> {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        if "safe" in response.lower():
            return {"safe": True, "violations": []}
        else:
            # Parse violation categories
            lines = response.split("\n")
            categories = lines[1].split(",") if len(lines) > 1 else []
            return {"safe": False, "violations": categories}
```

#### Pros:
- ✅ No API calls (privacy-friendly)
- ✅ Fast inference (8B model)
- ✅ Meta backing
- ✅ Multilingual

#### Cons:
- ❌ Requires GPU (adds infrastructure cost)
- ❌ Only safety categories (no PII, code safety)
- ❌ Less granular than Guardrails AI

#### Pricing:
- **Free** (self-hosted)
- **Infrastructure**: ~$0.50/hour for GPU instance (AWS g4dn.xlarge)

---

### 4. **Azure Content Safety**

**Website**: https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety  
**License**: Commercial (Azure)

#### Features:
- ✅ **Multi-modal** (text, images, videos)
- ✅ **4 categories** (hate, sexual, violence, self-harm)
- ✅ **Prompt shields** (jailbreak detection)
- ✅ **Custom categories** (train your own)
- ✅ **99.9% SLA**
- ✅ **SOC 2, ISO 27001 certified**

#### Code Example:
```python
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

# Initialize client
endpoint = "https://<resource>.cognitiveservices.azure.com"
credential = AzureKeyCredential("<api_key>")
client = ContentSafetyClient(endpoint, credential)

# Analyze text
from azure.ai.contentsafety.models import AnalyzeTextOptions

request = AnalyzeTextOptions(
    text="How do I create a virus?"
)

response = client.analyze_text(request)

# Check results
for category in response.categories_analysis:
    if category.severity > 2:  # 0-6 scale
        print(f"Violation: {category.category} (severity: {category.severity})")
```

#### Pros:
- ✅ Enterprise-grade (99.9% SLA)
- ✅ Compliance certifications
- ✅ Multi-modal (text + images)
- ✅ Microsoft support

#### Cons:
- ❌ Cloud-only (vendor lock-in)
- ❌ Costly ($1-$2 per 1K requests)
- ❌ Limited to safety categories

#### Pricing:
- **Standard**: $1.00 per 1K text records
- **Custom**: $1.50 per 1K text records

---

### 5. **AWS Bedrock Guardrails**

**Website**: https://aws.amazon.com/bedrock/guardrails/  
**License**: Commercial (AWS)

#### Features:
- ✅ **Content filters** (hate, violence, sexual, etc.)
- ✅ **PII redaction** (automatic)
- ✅ **Topic filtering** (denied topics)
- ✅ **Word filters** (custom blocklists)
- ✅ **Contextual grounding** (hallucination detection)
- ✅ **Integrated with Bedrock LLMs**

#### Code Example:
```python
import boto3

bedrock = boto3.client('bedrock-runtime')

# Invoke model with guardrails
response = bedrock.invoke_model(
    modelId='anthropic.claude-v2',
    guardrailIdentifier='<guardrail-id>',
    guardrailVersion='1',
    body={
        'prompt': 'Generate a test for this code...',
        'max_tokens': 1000
    }
)

# Response includes guardrail results
print(response['guardrailResults'])
```

#### Pros:
- ✅ Native AWS integration
- ✅ PII redaction built-in
- ✅ Contextual grounding (hallucination detection)
- ✅ AWS support

#### Cons:
- ❌ AWS-only (vendor lock-in)
- ❌ Costly
- ❌ Requires Bedrock

#### Pricing:
- **Content filters**: $0.75 per 1K requests
- **PII redaction**: $1.00 per 1K requests
- **Contextual grounding**: $1.50 per 1K requests

---

### 6. **Microsoft Presidio** (PII Detection)

**Website**: https://microsoft.github.io/presidio/  
**GitHub**: https://github.com/microsoft/presidio  
**License**: MIT

#### Features:
- ✅ **PII detection** (50+ entity types)
- ✅ **Multi-language** (20+ languages)
- ✅ **Custom recognizers**
- ✅ **Anonymization/pseudonymization**
- ✅ **Image redaction** (OCR + PII detection)
- ✅ **Structured data support** (JSON, CSV)

#### Code Example:
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Analyze text
text = "My email is john@example.com and SSN is 123-45-6789"
results = analyzer.analyze(
    text=text,
    language='en',
    entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]
)

# Anonymize
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# Output: "My email is <EMAIL_ADDRESS> and SSN is <US_SSN>"
```

#### Integration:
```python
# src/guardrails/presidio_integration.py
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PresidioPIIGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
    
    def detect_pii(self, text: str) -> list:
        results = self.analyzer.analyze(text=text, language='en')
        return [
            {
                "type": r.entity_type,
                "text": text[r.start:r.end],
                "score": r.score
            }
            for r in results
        ]
    
    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language='en')
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
```

#### Pros:
- ✅ Best-in-class PII detection
- ✅ Open source (MIT license)
- ✅ Microsoft backing
- ✅ Multi-language support

#### Cons:
- ❌ Only PII (no prompt injection, toxicity)
- ❌ Requires additional libraries for full coverage

#### Pricing:
- **Free** (self-hosted)

---

### 7. **LangKit** (WhyLabs)

**Website**: https://github.com/whylabs/langkit  
**License**: Apache 2.0

#### Features:
- ✅ **LLM observability** (metrics, logging)
- ✅ **Guardrails** (toxicity, PII, prompt injection)
- ✅ **Hallucination detection**
- ✅ **Cost tracking**
- ✅ **Drift detection** (model behavior changes)
- ✅ **WhyLabs integration** (monitoring dashboard)

#### Code Example:
```python
from langkit import llm_metrics, extract

# Define metrics to track
schema = llm_metrics.init()

# Analyze text
text = "My email is john@example.com"
profile = extract({"prompt": text, "response": "..."}, schema=schema)

# Check for issues
if profile["prompt.has_patterns"]:
    print("PII detected!")
```

#### Pros:
- ✅ Combined guardrails + observability
- ✅ Cost tracking built-in
- ✅ Drift detection (unique feature)

#### Cons:
- ❌ Less mature than Guardrails AI
- ❌ Requires WhyLabs for full features

#### Pricing:
- **Open Source**: Free
- **WhyLabs Cloud**: $500-$5000/month

---

## 📊 Comparison Matrix

| Library | Input Guards | Output Guards | PII | Prompt Injection | Code Safety | HITL | Budget | Audit | Cost | Best For |
|---------|--------------|---------------|-----|------------------|-------------|------|--------|-------|------|----------|
| **Guardrails AI** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Free-$999/mo | **General purpose** ⭐ |
| **NeMo Guardrails** | ✅ | ✅ | ⚠️ | ✅ | ❌ | ❌ | ❌ | ❌ | Free | **Conversational AI** |
| **LLaMA Guard 2** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | GPU cost | **Privacy-first** |
| **Azure Content Safety** | ✅ | ✅ | ⚠️ | ✅ | ❌ | ❌ | ❌ | ❌ | $1-2/1K | **Enterprise Azure** |
| **AWS Bedrock** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | $0.75-1.5/1K | **Enterprise AWS** |
| **Presidio** | ✅ | ✅ | ✅✅ | ❌ | ❌ | ❌ | ❌ | ❌ | Free | **PII detection only** |
| **LangKit** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | $500+/mo | **Observability focus** |
| **Custom (Current)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Dev time | **Full control** |

**Legend**: ✅ = Full support, ⚠️ = Partial support, ❌ = Not supported

---

## 🎯 Recommended Approach: **Hybrid Architecture**

Instead of choosing one, use a **layered approach**:

```python
# src/guardrails/hybrid_guard_manager.py

from guardrails import Guard
from guardrails.hub import PII, ToxicLanguage, PromptInjection, CodeSafety
from presidio_analyzer import AnalyzerEngine
from src.guardrails.policy_engine import PolicyEngine
from src.guardrails.hitl_manager import HITLManager
from src.guardrails.budget_tracker import BudgetTracker
from src.guardrails.audit_logger import AuditLogger

class HybridGuardManager:
    """
    Hybrid guardrail system combining:
    - Guardrails AI: Input/output validation
    - Presidio: PII detection
    - Custom: Policy, HITL, budget, audit
    """
    
    def __init__(self, session_id: str):
        # Layer 1: Guardrails AI (input/output validation)
        self.input_guard = Guard().use_many(
            PromptInjection(threshold=0.9),
            ToxicLanguage(threshold=0.8)
        )
        
        self.output_guard = Guard().use_many(
            CodeSafety(dangerous_patterns=["eval", "exec", "os.system"]),
            PII(pii_entities=["EMAIL", "PHONE", "SSN"])
        )
        
        # Layer 2: Presidio (advanced PII)
        self.pii_analyzer = AnalyzerEngine()
        
        # Layer 3: Custom (domain logic)
        self.policy_engine = PolicyEngine()
        self.hitl_manager = HITLManager()
        self.budget_tracker = BudgetTracker(session_id)
        self.audit_logger = AuditLogger()
    
    def check_input(self, text: str, context: dict = None) -> dict:
        """Layered input validation."""
        
        # Layer 1: Guardrails AI (fast checks)
        gr_result = self.input_guard.validate(text)
        if not gr_result.validation_passed:
            self.audit_logger.log_violation("guardrails_ai", gr_result.error)
            return {"safe": False, "reason": "Guardrails AI violation"}
        
        # Layer 2: Presidio (deep PII scan)
        pii_results = self.pii_analyzer.analyze(text=text, language='en')
        if pii_results:
            self.audit_logger.log_violation("presidio_pii", pii_results)
            return {"safe": False, "reason": "PII detected"}
        
        # Layer 3: Custom policy
        policy_result = self.policy_engine.evaluate("input_check", {}, context)
        if policy_result.decision == "DENY":
            return {"safe": False, "reason": policy_result.reason}
        
        # Layer 4: Budget check
        if not self.budget_tracker.check_budget("tokens", estimated_tokens=100):
            return {"safe": False, "reason": "Budget exceeded"}
        
        return {"safe": True, "sanitized": gr_result.validated_output}
    
    def check_output(self, code: str, context: dict = None) -> dict:
        """Layered output validation."""
        
        # Layer 1: Guardrails AI (code safety)
        gr_result = self.output_guard.validate(code)
        if not gr_result.validation_passed:
            self.audit_logger.log_violation("code_safety", gr_result.error)
            return {"safe": False, "reason": "Code safety violation"}
        
        # Layer 2: Custom policy
        policy_result = self.policy_engine.evaluate("output_check", {}, context)
        if policy_result.decision == "DENY":
            return {"safe": False, "reason": policy_result.reason}
        
        # Layer 3: HITL (if high-risk)
        if policy_result.decision == "REVIEW":
            approval = self.hitl_manager.request_approval(...)
            if not approval.approved:
                return {"safe": False, "reason": "Human denied"}
        
        return {"safe": True, "validated_code": gr_result.validated_output}
```

### Benefits of Hybrid Approach:

1. **Best of Both Worlds**:
   - Library: Fast, proven input/output validation
   - Custom: Domain-specific policies, HITL, budget

2. **Cost Optimization**:
   - Use free/cheap libraries for commodity checks
   - Invest dev time in unique business logic

3. **Vendor Independence**:
   - Can swap Guardrails AI for NeMo without rewriting
   - Custom layer remains stable

4. **Compliance**:
   - Libraries handle PII, toxicity (proven patterns)
   - Custom layer ensures audit trails, HITL for compliance

---

## 💰 Cost Analysis

### Current (Manual):
- **Development**: 40-80 hours (~$8K-$16K)
- **Maintenance**: 5-10 hours/month (~$1K-$2K/mo)
- **Total Year 1**: ~$20K-$40K

### Guardrails AI (Recommended):
- **Setup**: 8-16 hours (~$1.6K-$3.2K)
- **License**: $0-$999/month
- **Maintenance**: 2-4 hours/month (~$400-$800/mo)
- **Total Year 1**: ~$8K-$16K (50% savings)

### Azure/AWS (Enterprise):
- **Setup**: 16-24 hours (~$3.2K-$4.8K)
- **API Costs**: $500-$5K/month (depends on volume)
- **Maintenance**: 2-4 hours/month (~$400-$800/mo)
- **Total Year 1**: ~$10K-$65K

---

## 🎯 Final Recommendation

### **For AgenticTestGenerator:**

**Recommended Stack**:
1. **Guardrails AI** - Input/output validation ($0-$99/mo)
2. **Microsoft Presidio** - PII detection (free)
3. **Keep Custom** - Policy engine, HITL, budget, audit

**Implementation Priority**:
```python
# Week 1: Integrate Guardrails AI
pip install guardrails-ai
# Replace InputGuardrails, OutputGuardrails with Guardrails AI

# Week 2: Integrate Presidio
pip install presidio-analyzer presidio-anonymizer
# Enhance PII detection

# Week 3: Test & Validate
# Run security testing, compare coverage

# Week 4: Production rollout
# Enable hybrid guardrails, monitor metrics
```

**Expected Outcomes**:
- ✅ **50% less code** to maintain
- ✅ **Better security** (proven patterns)
- ✅ **Faster updates** (library patches)
- ✅ **Lower costs** ($8K vs $20K Year 1)

---

## 📚 Resources

### Documentation:
- Guardrails AI: https://docs.guardrailsai.com
- NeMo Guardrails: https://docs.nvidia.com/nemo/guardrails/
- Presidio: https://microsoft.github.io/presidio/
- LLaMA Guard: https://huggingface.co/meta-llama/LlamaGuard-2-8b

### Installation:
```bash
# Guardrails AI
pip install guardrails-ai

# Presidio
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg

# NeMo Guardrails
pip install nemoguardrails

# LLaMA Guard (requires transformers)
pip install transformers torch
```

---

**Last Updated**: November 29, 2025  
**Version**: 1.0

