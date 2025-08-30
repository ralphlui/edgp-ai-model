"""
LLM Gateway for managing multiple LLM providers.
Supports OpenAI, Anthropic, AWS Bedrock, and local models.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from abc import ABC, abstractmethod
import logging
import boto3
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_core.callbacks import BaseCallbackHandler

from .config import Config

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    LOCAL = "local"


class BedrockModel(Enum):
    """AWS Bedrock model identifiers."""
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE = "amazon.titan-text-lite-v1"
    LLAMA2_13B = "meta.llama2-13b-chat-v1"
    LLAMA2_70B = "meta.llama2-70b-chat-v1"


class LLMResponse:
    """Standardized response from LLM providers."""
    
    def __init__(
        self,
        content: str,
        provider: LLMProvider,
        model: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.cost = cost
        self.latency_ms = latency_ms
        self.metadata = metadata or {}


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate responses from the LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = ChatOpenAI(
            api_key=config.get("api_key"),
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", 0.7)
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await self.client.ainvoke(messages)
        
        return LLMResponse(
            content=response.content,
            provider=LLMProvider.OPENAI,
            model=self.config.get("model", "gpt-4")
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate using OpenAI."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        async for chunk in self.client.astream(messages):
            yield chunk.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = ChatAnthropic(
            api_key=config.get("api_key"),
            model=config.get("model", "claude-3-sonnet-20240229")
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await self.client.ainvoke(messages)
        
        return LLMResponse(
            content=response.content,
            provider=LLMProvider.ANTHROPIC,
            model=self.config.get("model", "claude-3-sonnet-20240229")
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate using Anthropic."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        async for chunk in self.client.astream(messages):
            yield chunk.content


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize Bedrock session
        session = boto3.Session(
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            region_name=config.get("aws_region", "us-east-1")
        )
        
        # Determine if we should use chat or completion model
        model_id = config.get("model_id", BedrockModel.CLAUDE_3_SONNET.value)
        
        if "claude" in model_id.lower():
            self.client = ChatBedrock(
                model_id=model_id,
                region_name=config.get("aws_region", "us-east-1"),
                credentials_profile_name=config.get("aws_profile"),
                model_kwargs={
                    "temperature": config.get("temperature", 0.7),
                    "max_tokens": config.get("max_tokens", 1000)
                }
            )
        else:
            self.client = BedrockLLM(
                model_id=model_id,
                region_name=config.get("aws_region", "us-east-1"),
                credentials_profile_name=config.get("aws_profile"),
                model_kwargs={
                    "temperature": config.get("temperature", 0.7),
                    "maxTokenCount": config.get("max_tokens", 1000)
                }
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using AWS Bedrock."""
        if isinstance(self.client, ChatBedrock):
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = await self.client.ainvoke(messages)
            content = response.content
        else:
            # For non-chat models, combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = await self.client.ainvoke(full_prompt)
            content = response
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.BEDROCK,
            model=self.config.get("model_id", "bedrock-model")
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate using AWS Bedrock."""
        if isinstance(self.client, ChatBedrock):
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            async for chunk in self.client.astream(messages):
                yield chunk.content
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            async for chunk in self.client.astream(full_prompt):
                yield chunk


class LLMGateway:
    """
    Centralized gateway for managing multiple LLM providers.
    Provides load balancing, failover, and unified interface.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize configured LLM providers."""
        llm_config = self.config.llm_providers
        
        # Initialize OpenAI
        if openai_config := llm_config.get("openai"):
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(openai_config)
            self.logger.info("Initialized OpenAI provider")
        
        # Initialize Anthropic
        if anthropic_config := llm_config.get("anthropic"):
            self.providers[LLMProvider.ANTHROPIC] = AnthropicProvider(anthropic_config)
            self.logger.info("Initialized Anthropic provider")
        
        # Initialize AWS Bedrock
        if bedrock_config := llm_config.get("bedrock"):
            self.providers[LLMProvider.BEDROCK] = BedrockProvider(bedrock_config)
            self.logger.info("Initialized AWS Bedrock provider")
        
        if not self.providers:
            raise ValueError("No LLM providers configured")
    
    async def generate(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using specified or default provider.
        
        Args:
            prompt: User prompt
            provider: Specific provider to use (optional)
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse: Standardized response object
        """
        target_provider = provider or self._get_default_provider()
        
        if target_provider not in self.providers:
            raise ValueError(f"Provider {target_provider} not available")
        
        try:
            return await self.providers[target_provider].generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error with provider {target_provider}: {e}")
            # Implement fallback logic here
            return await self._fallback_generate(
                prompt, system_prompt, temperature, max_tokens, **kwargs
            )
    
    async def stream_generate(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generate responses."""
        target_provider = provider or self._get_default_provider()
        
        if target_provider not in self.providers:
            raise ValueError(f"Provider {target_provider} not available")
        
        async for chunk in self.providers[target_provider].stream_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    def _get_default_provider(self) -> LLMProvider:
        """Get the default provider based on availability and configuration."""
        # Priority order: Bedrock -> OpenAI -> Anthropic -> Local
        if LLMProvider.BEDROCK in self.providers:
            return LLMProvider.BEDROCK
        elif LLMProvider.OPENAI in self.providers:
            return LLMProvider.OPENAI
        elif LLMProvider.ANTHROPIC in self.providers:
            return LLMProvider.ANTHROPIC
        else:
            return list(self.providers.keys())[0]
    
    async def _fallback_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Implement fallback logic when primary provider fails."""
        for provider_enum, provider in self.providers.items():
            try:
                self.logger.info(f"Trying fallback provider: {provider_enum}")
                return await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Fallback provider {provider_enum} also failed: {e}")
                continue
        
        raise Exception("All LLM providers failed")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get information about a specific provider."""
        if provider not in self.providers:
            return {}
        
        return {
            "provider": provider.value,
            "available": True,
            "config": {k: v for k, v in self.providers[provider].config.items() 
                      if k not in ["api_key", "aws_secret_access_key"]}
        }
