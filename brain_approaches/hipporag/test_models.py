"""
Test free models from free-models.yml to check which ones are active (no credit errors).

Updates free-models.yml with status: "active" or "inactive" based on test results.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger

from src.config import get_config
from src.utils.openrouter_client import OpenRouterClient

from .config import FREE_MODELS_YAML


def _is_credit_error(error: str) -> bool:
    """Check if error message indicates a credit/affordability issue."""
    if not error:
        return False
    error_lower = error.lower()
    credit_keywords = ["credit", "afford", "insufficient", "requires more credits", "can only afford"]
    return any(keyword in error_lower for keyword in credit_keywords)


async def test_model(model_id: str, api_key: str) -> tuple[bool, str]:
    """
    Test a single model with a simple prompt.

    Returns:
        Tuple of (is_active, error_message). is_active=True if model works, False if credit error.
    """
    test_prompt = "Extract triples from: 'Alice works at Google. Bob is a friend of Alice.'"
    system_prompt = "You are a helpful assistant that extracts knowledge triples."

    try:
        async with OpenRouterClient(api_key=api_key, model=model_id) as client:
            response = await client.generate(
                prompt=test_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=100,
            )
            if response and len(response) > 0:
                logger.info("Model '{}' is ACTIVE (test successful)", model_id)
                return True, ""
            else:
                logger.warning("Model '{}' returned empty response", model_id)
                return False, "Empty response"
    except Exception as e:
        error_msg = str(e)
        is_credit_error = _is_credit_error(error_msg)
        if is_credit_error:
            logger.warning("Model '{}' is INACTIVE (credit error): {}", model_id, error_msg[:200])
            return False, f"Credit error: {error_msg[:200]}"
        else:
            logger.warning("Model '{}' failed with non-credit error: {}", model_id, error_msg[:200])
            return False, f"Error: {error_msg[:200]}"


async def test_all_models() -> None:
    """Test all models in free-models.yml and update their status."""
    if not FREE_MODELS_YAML.exists():
        logger.error("free-models.yml not found at {}", FREE_MODELS_YAML)
        return

    # Load models
    with FREE_MODELS_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if not models:
        logger.error("No models found in free-models.yml")
        return

    logger.info("Testing {} models from free-models.yml", len(models))

    # Get API key
    cfg = get_config()
    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        return

    # Test each model
    for model in models:
        model_id = model.get("id")
        if not model_id:
            continue

        logger.info("Testing model: {} ({})", model_id, model.get("name", "unknown"))
        is_active, error_msg = await test_model(model_id, api_key)

        # Update status
        model["status"] = "active" if is_active else "inactive"
        if error_msg:
            model["last_error"] = error_msg
        elif "last_error" in model:
            del model["last_error"]

        # Small delay between tests
        await asyncio.sleep(1)

    # Save updated YAML
    with FREE_MODELS_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, indent=2, default_flow_style=False, sort_keys=False)

    # Print summary
    active_count = sum(1 for m in models if m.get("status") == "active")
    inactive_count = len(models) - active_count
    logger.info(
        "Model testing complete: {} active, {} inactive out of {} total models",
        active_count,
        inactive_count,
        len(models),
    )

    print("\n" + "=" * 70)
    print("MODEL TEST SUMMARY")
    print("=" * 70)
    for model in models:
        status = model.get("status", "unknown")
        status_icon = "✅" if status == "active" else "❌"
        print(f"{status_icon} {model.get('id')}: {status}")
        if status == "inactive" and "last_error" in model:
            print(f"   Error: {model['last_error'][:100]}")
    print("=" * 70)


def main() -> None:
    """Entry point for model testing."""
    asyncio.run(test_all_models())


if __name__ == "__main__":
    main()
