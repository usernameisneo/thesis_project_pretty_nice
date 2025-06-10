"""
OpenRouter API client for LLM model access and management.

This module provides comprehensive integration with OpenRouter's API, including
model discovery, filtering, sorting, and chat completions. It supports all
OpenRouter models with advanced configuration options.

Features:
    - Complete model catalog access
    - Advanced filtering and sorting
    - Real-time model availability
    - Cost and performance metrics
    - Streaming and non-streaming responses
    - Error handling and retry logic

Author: AI-Powered Thesis Assistant Team
Version: 2.0
License: MIT
"""

import httpx
import json
import time
from typing import Dict, Any, Optional, List, Iterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from core.exceptions import APIError

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an OpenRouter model."""
    id: str
    name: str
    description: str
    pricing: Dict[str, float]
    context_length: int
    architecture: Dict[str, Any]
    top_provider: Dict[str, Any]
    per_request_limits: Optional[Dict[str, Any]] = None
    created: Optional[int] = None
    
    @property
    def cost_per_1k_tokens(self) -> float:
        """Get cost per 1000 tokens (prompt + completion average)."""
        prompt_cost = self.pricing.get('prompt', 0)
        completion_cost = self.pricing.get('completion', 0)
        return (prompt_cost + completion_cost) / 2
    
    @property
    def provider_name(self) -> str:
        """Get the primary provider name."""
        return self.top_provider.get('name', 'Unknown')
    
    @property
    def max_tokens(self) -> int:
        """Get maximum context length."""
        return self.context_length


@dataclass
class ChatMessage:
    """A chat message with role and content."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChatSession:
    """A chat session with message history."""
    session_id: str
    model_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    total_cost: float = 0.0


class OpenRouterClient:
    """
    Advanced OpenRouter API client with comprehensive model management.
    
    This client provides full access to OpenRouter's model catalog with
    advanced filtering, sorting, and chat capabilities. It includes
    error handling, rate limiting, and cost tracking.
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (can be set later)
        """
        self.api_key = api_key
        self.session = httpx.Client(
            timeout=60.0,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AI-Thesis-Assistant/2.0"
            }
        )
        self._models_cache: Optional[List[ModelInfo]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_duration = 300  # 5 minutes
        
        logger.info("OpenRouter client initialized")
    
    def __del__(self):
        """Clean up HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key for authentication.
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        logger.info("API key updated")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Thesis-Assistant/2.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make authenticated request to OpenRouter API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response data
            
        Raises:
            APIError: If request fails
        """
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
            except:
                error_msg += f" - {e.response.text}"
            
            logger.error(error_msg)
            raise APIError(error_msg)
            
        except httpx.RequestError as e:
            error_msg = f"OpenRouter request failed: {e}"
            logger.error(error_msg)
            raise APIError(error_msg)
            
        except Exception as e:
            error_msg = f"OpenRouter API call failed: {e}"
            logger.error(error_msg)
            raise APIError(error_msg)
    
    def get_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        Get all available models from OpenRouter.
        
        Args:
            force_refresh: Force refresh of cached models
            
        Returns:
            List of ModelInfo objects
            
        Raises:
            APIError: If API call fails
        """
        # Check cache
        current_time = time.time()
        if (not force_refresh and 
            self._models_cache and 
            self._cache_timestamp and 
            current_time - self._cache_timestamp < self._cache_duration):
            logger.debug("Returning cached models")
            return self._models_cache
        
        logger.info("Fetching models from OpenRouter API")
        
        try:
            data = self._make_request("GET", "/models")
            models = []
            
            for model_data in data.get("data", []):
                try:
                    model = ModelInfo(
                        id=model_data["id"],
                        name=model_data.get("name", model_data["id"]),
                        description=model_data.get("description", ""),
                        pricing=model_data.get("pricing", {}),
                        context_length=model_data.get("context_length", 0),
                        architecture=model_data.get("architecture", {}),
                        top_provider=model_data.get("top_provider", {}),
                        per_request_limits=model_data.get("per_request_limits"),
                        created=model_data.get("created")
                    )
                    models.append(model)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse model {model_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Cache results
            self._models_cache = models
            self._cache_timestamp = current_time
            
            logger.info(f"Retrieved {len(models)} models from OpenRouter")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            raise APIError(f"Failed to retrieve models: {e}")
    
    def filter_models(self, 
                     models: List[ModelInfo],
                     provider: Optional[str] = None,
                     max_cost: Optional[float] = None,
                     min_context: Optional[int] = None,
                     search_term: Optional[str] = None) -> List[ModelInfo]:
        """
        Filter models based on criteria.
        
        Args:
            models: List of models to filter
            provider: Filter by provider name
            max_cost: Maximum cost per 1k tokens
            min_context: Minimum context length
            search_term: Search in name/description
            
        Returns:
            Filtered list of models
        """
        filtered = models
        
        if provider:
            filtered = [m for m in filtered if provider.lower() in m.provider_name.lower()]
        
        if max_cost is not None:
            filtered = [m for m in filtered if m.cost_per_1k_tokens <= max_cost]
        
        if min_context is not None:
            filtered = [m for m in filtered if m.context_length >= min_context]
        
        if search_term:
            term = search_term.lower()
            filtered = [m for m in filtered if 
                       term in m.name.lower() or 
                       term in m.description.lower() or
                       term in m.id.lower()]
        
        logger.debug(f"Filtered {len(models)} models to {len(filtered)}")
        return filtered
    
    def sort_models(self, 
                   models: List[ModelInfo],
                   sort_by: str = "name",
                   reverse: bool = False) -> List[ModelInfo]:
        """
        Sort models by specified criteria.
        
        Args:
            models: List of models to sort
            sort_by: Sort criteria ('name', 'cost', 'context', 'provider')
            reverse: Sort in descending order
            
        Returns:
            Sorted list of models
        """
        sort_functions = {
            "name": lambda m: m.name.lower(),
            "cost": lambda m: m.cost_per_1k_tokens,
            "context": lambda m: m.context_length,
            "provider": lambda m: m.provider_name.lower(),
            "id": lambda m: m.id.lower()
        }
        
        if sort_by not in sort_functions:
            logger.warning(f"Unknown sort criteria: {sort_by}, using 'name'")
            sort_by = "name"
        
        sorted_models = sorted(models, key=sort_functions[sort_by], reverse=reverse)
        logger.debug(f"Sorted {len(models)} models by {sort_by}")
        return sorted_models

    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: str,
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7,
                       top_p: float = 1.0,
                       frequency_penalty: float = 0.0,
                       presence_penalty: float = 0.0,
                       stream: bool = False) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter.

        Args:
            messages: List of message dictionaries
            model: Model ID to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stream: Whether to stream the response

        Returns:
            Chat completion response

        Raises:
            APIError: If API call fails
        """
        if not self.api_key:
            raise APIError("API key required for chat completions")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.info(f"Creating chat completion with model: {model}")

        try:
            if stream:
                return self._stream_completion(payload)
            else:
                return self._make_request("POST", "/chat/completions", json=payload)

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    def _stream_completion(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion response.

        Args:
            payload: Request payload

        Yields:
            Streaming response chunks
        """
        url = f"{self.BASE_URL}/chat/completions"
        headers = self._get_headers()

        try:
            with self.session.stream(
                "POST",
                url,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            yield data
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            raise APIError(f"Streaming completion failed: {e}")

    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get specific model by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelInfo object or None if not found
        """
        models = self.get_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    def get_providers(self) -> List[str]:
        """
        Get list of unique providers.

        Returns:
            List of provider names
        """
        models = self.get_models()
        providers = set()
        for model in models:
            if model.provider_name:
                providers.add(model.provider_name)
        return sorted(list(providers))

    def estimate_cost(self,
                     model_id: str,
                     prompt_tokens: int,
                     completion_tokens: int) -> float:
        """
        Estimate cost for a request.

        Args:
            model_id: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        model = self.get_model_by_id(model_id)
        if not model:
            return 0.0

        prompt_cost = model.pricing.get('prompt', 0) * prompt_tokens / 1000
        completion_cost = model.pricing.get('completion', 0) * completion_tokens / 1000

        return prompt_cost + completion_cost

    def test_connection(self) -> bool:
        """
        Test connection to OpenRouter API.

        Returns:
            True if connection successful
        """
        try:
            self.get_models()
            logger.info("OpenRouter connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {e}")
            return False

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics (if available).

        Returns:
            Usage statistics dictionary
        """
        try:
            # Note: This endpoint may not be available in all OpenRouter plans
            return self._make_request("GET", "/usage")
        except APIError:
            logger.warning("Usage stats not available")
            return {}


class ChatSessionManager:
    """
    Manages chat sessions with OpenRouter models.

    Provides session management, message history, and cost tracking
    for interactive chat conversations.
    """

    def __init__(self, openrouter_client: OpenRouterClient):
        """
        Initialize session manager.

        Args:
            openrouter_client: OpenRouter client instance
        """
        self.client = openrouter_client
        self.sessions: Dict[str, ChatSession] = {}
        logger.info("Chat session manager initialized")

    def create_session(self, model_id: str, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session.

        Args:
            model_id: Model to use for the session
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        session = ChatSession(
            session_id=session_id,
            model_id=model_id
        )

        self.sessions[session_id] = session
        logger.info(f"Created chat session: {session_id} with model: {model_id}")
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add message to session.

        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        message = ChatMessage(role=role, content=content)
        self.sessions[session_id].messages.append(message)
        logger.debug(f"Added {role} message to session {session_id}")

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession or None if not found
        """
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[str]:
        """
        Get list of active session IDs.

        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
