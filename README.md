# Cache-Augmented Generation with LLaMA 2 (4-bit)

This notebook demonstrates how to use **Cache-Augmented Generation (CAG)** with a 4-bit quantized version of LLaMA 2 to answer questions from a fixed knowledge base. Instead of refeeding context with every prompt, the model preloads knowledge into its internal key-value (KV) cache for efficient and consistent responses.

## Overview

**Cache-Augmented Generation (CAG)** is a technique where the model first ingests a static knowledge base, storing it in memory via its KV cache. Later, user queries are answered in context of this cached knowledge.

### Benefits of CAG:
- Reduces redundant context prompts
- Speeds up inference
- Mimics "memory" for static datasets

## Setup

Install the required packages:
```python
pip install -U bitsandbytes transformers accelerate
```

Login to Hugging Face to access gated models:
```python
from huggingface_hub import notebook_login
notebook_login()
```

## Knowledge Base Example
We use a fictional dataset of tech device incidents:

```bash
Incident 1: Smartwatch Sync Issue
Device: TechFit Pro Smartwatch
Problem: Fails to sync with Android phones due to firmware bug.
Fix: Firmware update version 2.1.4 resolved Bluetooth sync issues.

Incident 2: Noise Cancelling Headphones Overheating
Device: SoundBliss NC700
Problem: Excessive heat during long use, especially when ANC is on.
Fix: Manufacturer advised users to limit use, offered refunds.

Incident 3: Fitness Band Step Miscount
Device: FitRun 360
Problem: Inaccurate step count, overestimates by 20–30%.
Fix: Patch 1.0.5 released to fix accelerometer calibration.

```

## How Cache-Augmented Generation Works

1. The model preloads the knowledge base using `use_cache=True`, storing it in the KV (key-value) cache.
2. During inference, only the user question is passed, and the model generates an answer using the retained cache.
3. This avoids repeating long prompts and improves response consistency.

## Example Prompts

Here are sample questions you can ask based on the cached knowledge:

```python
ask_question("List all devices mentioned in the incidents. Just names.", kv_cache)
ask_question("What was wrong with the TechFit Pro Smartwatch? Keep it brief.", kv_cache)
ask_question("What solution was offered for overheating headphones?", kv_cache)

```

## Sample Outputs
```bash
TechFit Pro Smartwatch, SoundBliss NC700, FitRun 360
Firmware bug caused syncing issues.
Manufacturer advised users to limit use, offered refunds.
```

## Disclaimer
This notebook uses fictional data and is intended for educational demonstration of Cache-Augmented Generation (CAG). Not suitable for production without further validation, testing, and safeguards.

## References & Credits

- [Meta AI – LLaMA](https://ai.meta.com/llama/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [bitsandbytes by Tim Dettmers](https://github.com/TimDettmers/bitsandbytes)


