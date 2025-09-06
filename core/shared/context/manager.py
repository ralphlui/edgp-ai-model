"""
Context Management System

Central management system for AI agent context including conversation history,
session state, user preferences, and shared context across agents.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import uuid

from .types import (
    Message, ConversationHistory, UserPreferences, SessionState, AgentContext,
    SharedContext, ContextConfig, ContextAnalytics, MessageRole, SessionStatus,
    ContextScope, ContextType, BaseContextStore
)
from .stores import ContextStoreManager

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and messaging."""
    
    def __init__(self, store_manager: ContextStoreManager):
        self.store_manager = store_manager
        self.active_conversations: Dict[str, ConversationHistory] = {}
    
    async def create_conversation(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> ConversationHistory:
        """Create a new conversation."""
        conversation = ConversationHistory(
            session_id=session_id,
            user_id=user_id,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Store in database
        await self.store_manager.store_conversation(conversation)
        
        # Cache active conversation
        self.active_conversations[conversation.conversation_id] = conversation
        
        logger.info("Created conversation %s for session %s", 
                   conversation.conversation_id, session_id)
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get conversation by ID."""
        # Check cache first
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Load from storage
        conversation = await self.store_manager.get_conversation(conversation_id)
        if conversation:
            self.active_conversations[conversation_id] = conversation
        
        return conversation
    
    async def add_message(
        self,
        conversation_id: str,
        message: Message
    ) -> bool:
        """Add a message to a conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Add message to conversation
        conversation.add_message(message)
        
        # Update in storage
        await self.store_manager.store_conversation(conversation)
        
        return True
    
    async def create_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        agent_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Create and add a message to conversation."""
        message = Message(
            role=role,
            content=content,
            agent_id=agent_id,
            tool_calls=tool_calls or [],
            metadata=metadata or {}
        )
        
        await self.add_message(conversation_id, message)
        return message
    
    async def get_conversation_by_session(self, session_id: str) -> List[ConversationHistory]:
        """Get all conversations for a session."""
        return await self.store_manager.get_conversations_by_session(session_id)
    
    async def search_conversations(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationHistory]:
        """Search conversations."""
        return await self.store_manager.search_conversations(user_id, query, limit)
    
    async def summarize_conversation(self, conversation_id: str) -> Optional[str]:
        """Generate conversation summary."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return None
        
        # Simple summarization - could be enhanced with AI
        user_messages = conversation.get_messages_by_role(MessageRole.USER)
        assistant_messages = conversation.get_messages_by_role(MessageRole.ASSISTANT)
        
        summary = f"Conversation with {len(user_messages)} user messages and {len(assistant_messages)} responses"
        
        if user_messages:
            first_message = user_messages[0].content[:100]
            summary += f". Started with: {first_message}..."
        
        conversation.summary = summary
        await self.store_manager.store_conversation(conversation)
        
        return summary


class SessionManager:
    """Manages user sessions and state."""
    
    def __init__(self, store_manager: ContextStoreManager, config: ContextConfig):
        self.store_manager = store_manager
        self.config = config
        self.active_sessions: Dict[str, SessionState] = {}
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager initialized")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        session_type: str = "user"
    ) -> SessionState:
        """Create a new session."""
        session = SessionState(
            user_id=user_id,
            session_type=session_type,
            expires_at=datetime.now() + timedelta(hours=self.config.session_timeout_hours)
        )
        
        # Store in database
        await self.store_manager.store_session(session)
        
        # Cache active session
        self.active_sessions[session.session_id] = session
        
        logger.info("Created session %s for user %s", session.session_id, user_id)
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        # Check cache first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session.is_expired():
                await self.terminate_session(session_id)
                return None
            return session
        
        # Load from storage
        session = await self.store_manager.get_session(session_id)
        if session:
            if session.is_expired():
                await self.terminate_session(session_id)
                return None
            self.active_sessions[session_id] = session
        
        return session
    
    async def update_session(self, session: SessionState) -> bool:
        """Update session state."""
        session.update_activity()
        
        # Update in storage
        success = await self.store_manager.store_session(session)
        
        # Update cache
        if success:
            self.active_sessions[session.session_id] = session
        
        return success
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a session."""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.TERMINATED
        session.update_activity()
        
        # Update in storage
        await self.store_manager.store_session(session)
        
        # Remove from cache
        self.active_sessions.pop(session_id, None)
        
        logger.info("Terminated session %s", session_id)
        return True
    
    async def extend_session(self, session_id: str, hours: int = None) -> bool:
        """Extend session expiration."""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        if hours is None:
            hours = self.config.session_timeout_hours
        
        session.expires_at = datetime.now() + timedelta(hours=hours)
        return await self.update_session(session)
    
    async def get_user_sessions(self, user_id: str) -> List[SessionState]:
        """Get all sessions for a user."""
        return await self.store_manager.get_user_sessions(user_id)
    
    async def _cleanup_loop(self):
        """Background cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Find expired sessions
                expired_sessions = []
                for session_id, session in list(self.active_sessions.items()):
                    if session.is_expired():
                        expired_sessions.append(session_id)
                
                # Terminate expired sessions
                for session_id in expired_sessions:
                    await self.terminate_session(session_id)
                
                # Clean up storage
                await self.store_manager.cleanup_expired()
                
                if expired_sessions:
                    logger.info("Cleaned up %d expired sessions", len(expired_sessions))
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in session cleanup: %s", str(e))
    
    async def shutdown(self):
        """Shutdown session manager."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SessionManager shutdown complete")


class UserPreferenceManager:
    """Manages user preferences and settings."""
    
    def __init__(self, store_manager: ContextStoreManager):
        self.store_manager = store_manager
        self.preference_cache: Dict[str, UserPreferences] = {}
    
    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating default if not exists."""
        # Check cache first
        if user_id in self.preference_cache:
            return self.preference_cache[user_id]
        
        # Load from storage
        preferences = await self.store_manager.get_user_preferences(user_id)
        
        if not preferences:
            # Create default preferences
            preferences = UserPreferences(user_id=user_id)
            await self.store_manager.store_user_preferences(preferences)
        
        # Cache preferences
        self.preference_cache[user_id] = preferences
        return preferences
    
    async def update_user_preferences(self, preferences: UserPreferences) -> bool:
        """Update user preferences."""
        success = await self.store_manager.store_user_preferences(preferences)
        
        if success:
            self.preference_cache[preferences.user_id] = preferences
        
        return success
    
    async def update_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Update a specific preference."""
        preferences = await self.get_user_preferences(user_id)
        preferences.update_preference(key, value)
        return await self.update_user_preferences(preferences)
    
    async def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a specific preference value."""
        preferences = await self.get_user_preferences(user_id)
        return preferences.get_preference(key, default)


class SharedContextManager:
    """Manages shared context across agents and sessions."""
    
    def __init__(self, store_manager: ContextStoreManager, config: ContextConfig):
        self.store_manager = store_manager
        self.config = config
        self.context_cache: Dict[str, SharedContext] = {}
    
    async def create_shared_context(
        self,
        name: str,
        scope: ContextScope,
        data: Optional[Dict[str, Any]] = None,
        owner_id: Optional[str] = None
    ) -> SharedContext:
        """Create a new shared context."""
        context = SharedContext(
            name=name,
            scope=scope,
            data=data or {},
            owner_id=owner_id
        )
        
        # Store in database
        await self.store_manager.store_shared_context(context)
        
        # Cache context
        self.context_cache[context.context_id] = context
        
        logger.info("Created shared context %s with scope %s", 
                   context.context_id, scope.value)
        return context
    
    async def get_shared_context(self, context_id: str) -> Optional[SharedContext]:
        """Get shared context by ID."""
        # Check cache first
        if context_id in self.context_cache:
            context = self.context_cache[context_id]
            if context.is_expired():
                await self.delete_shared_context(context_id)
                return None
            return context
        
        # Load from storage
        context = await self.store_manager.get_shared_context(context_id)
        if context:
            if context.is_expired():
                await self.delete_shared_context(context_id)
                return None
            self.context_cache[context_id] = context
        
        return context
    
    async def update_shared_context(
        self,
        context_id: str,
        key: str,
        value: Any,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Update shared context data."""
        context = await self.get_shared_context(context_id)
        if not context:
            return False
        
        # Check access permissions
        if not context.can_access(agent_id, user_id):
            logger.warning("Access denied to context %s for agent %s user %s", 
                          context_id, agent_id, user_id)
            return False
        
        # Update data
        context.update_data(key, value, agent_id)
        
        # Store updated context
        success = await self.store_manager.store_shared_context(context)
        
        if success:
            self.context_cache[context_id] = context
        
        return success
    
    async def get_shared_contexts(
        self,
        scope: Optional[ContextScope] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[SharedContext]:
        """Get shared contexts with filtering."""
        contexts = await self.store_manager.get_shared_contexts(scope)
        
        # Filter by access permissions
        accessible_contexts = []
        for context in contexts:
            if context.can_access(agent_id, user_id):
                accessible_contexts.append(context)
        
        return accessible_contexts
    
    async def delete_shared_context(self, context_id: str) -> bool:
        """Delete a shared context."""
        success = await self.store_manager.delete_shared_context(context_id)
        
        if success:
            self.context_cache.pop(context_id, None)
        
        return success


class AgentContextManager:
    """Manages agent-specific context."""
    
    def __init__(self, store_manager: ContextStoreManager):
        self.store_manager = store_manager
        self.agent_contexts: Dict[str, AgentContext] = {}
    
    async def get_agent_context(self, agent_id: str, session_id: str) -> AgentContext:
        """Get or create agent context for session."""
        context_key = f"{agent_id}:{session_id}"
        
        if context_key in self.agent_contexts:
            return self.agent_contexts[context_key]
        
        # Try to load from storage
        context = await self.store_manager.get_agent_context(agent_id, session_id)
        
        if not context:
            # Create new context
            context = AgentContext(
                agent_id=agent_id,
                session_id=session_id
            )
            await self.store_manager.store_agent_context(context)
        
        self.agent_contexts[context_key] = context
        return context
    
    async def update_agent_context(self, context: AgentContext) -> bool:
        """Update agent context."""
        success = await self.store_manager.store_agent_context(context)
        
        if success:
            context_key = f"{context.agent_id}:{context.session_id}"
            self.agent_contexts[context_key] = context
        
        return success
    
    async def update_agent_task(self, agent_id: str, session_id: str, task: str) -> bool:
        """Update agent's current task."""
        context = await self.get_agent_context(agent_id, session_id)
        context.update_task(task)
        return await self.update_agent_context(context)
    
    async def add_agent_goal(self, agent_id: str, session_id: str, goal: str) -> bool:
        """Add a goal to agent context."""
        context = await self.get_agent_context(agent_id, session_id)
        context.add_goal(goal)
        return await self.update_agent_context(context)


class ContextManager:
    """Central context management system."""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        
        # Component managers
        self.store_manager = ContextStoreManager(config)
        self.conversation_manager = ConversationManager(self.store_manager)
        self.session_manager = SessionManager(self.store_manager, config)
        self.preference_manager = UserPreferenceManager(self.store_manager)
        self.shared_context_manager = SharedContextManager(self.store_manager, config)
        self.agent_context_manager = AgentContextManager(self.store_manager)
        
        # Analytics
        self.analytics = ContextAnalytics()
        
        # Background tasks
        self._analytics_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the context manager."""
        await self.store_manager.initialize()
        await self.session_manager.initialize()
        
        # Start analytics collection
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        logger.info("ContextManager initialized")
    
    # Session management
    async def create_session(self, user_id: Optional[str] = None) -> SessionState:
        """Create a new session."""
        return await self.session_manager.create_session(user_id)
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return await self.session_manager.get_session(session_id)
    
    # Conversation management
    async def create_conversation(self, session_id: str, user_id: Optional[str] = None) -> ConversationHistory:
        """Create a new conversation."""
        return await self.conversation_manager.create_conversation(session_id, user_id)
    
    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Message:
        """Add a message to conversation."""
        return await self.conversation_manager.create_message(
            conversation_id, role, content, agent_id, **kwargs
        )
    
    # User preferences
    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences."""
        return await self.preference_manager.get_user_preferences(user_id)
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Update user preference."""
        return await self.preference_manager.update_preference(user_id, key, value)
    
    # Shared context
    async def create_shared_context(
        self,
        name: str,
        scope: ContextScope,
        data: Optional[Dict[str, Any]] = None
    ) -> SharedContext:
        """Create shared context."""
        return await self.shared_context_manager.create_shared_context(name, scope, data)
    
    async def update_shared_context(
        self,
        context_id: str,
        key: str,
        value: Any,
        agent_id: Optional[str] = None
    ) -> bool:
        """Update shared context."""
        return await self.shared_context_manager.update_shared_context(
            context_id, key, value, agent_id
        )
    
    # Agent context
    async def get_agent_context(self, agent_id: str, session_id: str) -> AgentContext:
        """Get agent context."""
        return await self.agent_context_manager.get_agent_context(agent_id, session_id)
    
    # Analytics
    async def get_analytics(self) -> ContextAnalytics:
        """Get context analytics."""
        return self.analytics
    
    async def _analytics_loop(self):
        """Background analytics collection."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self._update_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in analytics loop: %s", str(e))
    
    async def _update_analytics(self):
        """Update analytics data."""
        # Get counts from storage
        self.analytics.total_sessions = await self.store_manager.count_sessions()
        self.analytics.active_sessions = len(self.session_manager.active_sessions)
        self.analytics.total_conversations = await self.store_manager.count_conversations()
        self.analytics.total_messages = await self.store_manager.count_messages()
        self.analytics.shared_contexts = await self.store_manager.count_shared_contexts()
        
        self.analytics.last_updated = datetime.now()
    
    async def shutdown(self):
        """Shutdown the context manager."""
        # Cancel analytics task
        if self._analytics_task and not self._analytics_task.done():
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        await self.session_manager.shutdown()
        await self.store_manager.shutdown()
        
        logger.info("ContextManager shutdown complete")
