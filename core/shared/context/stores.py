"""
Context Storage Implementations

Storage backends for context data including sessions, conversations,
user preferences, and shared context.
"""

import asyncio
import json
import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiosqlite

from .types import (
    SessionState, ConversationHistory, UserPreferences, SharedContext,
    AgentContext, ContextConfig, BaseContextStore, Message, MessageRole,
    SessionStatus, ContextScope
)

logger = logging.getLogger(__name__)


class SQLiteContextStore(BaseContextStore):
    """SQLite-based context storage implementation."""
    
    def __init__(self, config: ContextConfig):
        super().__init__(config)
        self.db_path = Path(config.storage_path) / "context.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the SQLite database."""
        if self._initialized:
            return
        
        async with aiosqlite.connect(str(self.db_path)) as db:
            await self._create_tables(db)
            await db.commit()
        
        self._initialized = True
        logger.info("SQLiteContextStore initialized at %s", self.db_path)
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables."""
        # Sessions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                status TEXT NOT NULL,
                session_type TEXT,
                state_data TEXT,
                variables TEXT,
                flags TEXT,
                current_conversation_id TEXT,
                active_agents TEXT,
                current_workflow_id TEXT,
                total_conversations INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                total_duration REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                expires_at TEXT
            )
        """)
        
        # Conversations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                title TEXT,
                summary TEXT,
                tags TEXT,
                agent_ids TEXT,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                user_messages INTEGER DEFAULT 0,
                assistant_messages INTEGER DEFAULT 0,
                started_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                ended_at TEXT
            )
        """)
        
        # Messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_id TEXT,
                tool_calls TEXT,
                function_call TEXT,
                thread_id TEXT,
                tokens_used INTEGER,
                processing_time REAL,
                model_used TEXT,
                metadata TEXT,
                attachments TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
        """)
        
        # User preferences table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_language TEXT,
                communication_style TEXT,
                detail_level TEXT,
                theme TEXT,
                timezone TEXT,
                date_format TEXT,
                preferred_agents TEXT,
                agent_personalities TEXT,
                data_sharing BOOLEAN,
                analytics_opt_in BOOLEAN,
                personalization BOOLEAN,
                custom_settings TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Shared context table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS shared_contexts (
                context_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                scope TEXT NOT NULL,
                data TEXT,
                metadata TEXT,
                owner_id TEXT,
                allowed_agents TEXT,
                allowed_users TEXT,
                version INTEGER DEFAULT 1,
                revision_history TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            )
        """)
        
        # Agent contexts table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_contexts (
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                current_task TEXT,
                task_history TEXT,
                goals TEXT,
                working_memory TEXT,
                facts TEXT,
                assumptions TEXT,
                available_tools TEXT,
                tool_usage_history TEXT,
                performance_metrics TEXT,
                error_history TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, session_id)
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations (session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_shared_contexts_scope ON shared_contexts (scope)")
    
    async def store_session(self, session: SessionState) -> bool:
        """Store session state."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO sessions (
                        session_id, user_id, status, session_type, state_data,
                        variables, flags, current_conversation_id, active_agents,
                        current_workflow_id, total_conversations, total_messages,
                        total_duration, created_at, last_activity, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.status.value,
                    session.session_type,
                    json.dumps(session.state_data),
                    json.dumps(session.variables),
                    json.dumps(session.flags),
                    session.current_conversation_id,
                    json.dumps(session.active_agents),
                    session.current_workflow_id,
                    session.total_conversations,
                    session.total_messages,
                    session.total_duration,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.expires_at.isoformat() if session.expires_at else None
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store session %s: %s", session.session_id, str(e))
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session state."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_session(row)
                return None
        except Exception as e:
            logger.error("Failed to get session %s: %s", session_id, str(e))
            return None
    
    async def store_conversation(self, conversation: ConversationHistory) -> bool:
        """Store conversation history."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                # Store conversation metadata
                await db.execute("""
                    INSERT OR REPLACE INTO conversations (
                        conversation_id, session_id, user_id, title, summary,
                        tags, agent_ids, message_count, total_tokens,
                        user_messages, assistant_messages, started_at,
                        last_activity, ended_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation.conversation_id,
                    conversation.session_id,
                    conversation.user_id,
                    conversation.title,
                    conversation.summary,
                    json.dumps(conversation.tags),
                    json.dumps(conversation.agent_ids),
                    conversation.message_count,
                    conversation.total_tokens,
                    conversation.user_messages,
                    conversation.assistant_messages,
                    conversation.started_at.isoformat(),
                    conversation.last_activity.isoformat(),
                    conversation.ended_at.isoformat() if conversation.ended_at else None
                ))
                
                # Store messages
                for message in conversation.messages:
                    await db.execute("""
                        INSERT OR REPLACE INTO messages (
                            message_id, conversation_id, session_id, role, content,
                            agent_id, tool_calls, function_call, thread_id,
                            tokens_used, processing_time, model_used, metadata,
                            attachments, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        message.message_id,
                        message.conversation_id,
                        message.session_id,
                        message.role.value,
                        message.content,
                        message.agent_id,
                        json.dumps(message.tool_calls),
                        json.dumps(message.function_call) if message.function_call else None,
                        message.thread_id,
                        message.tokens_used,
                        message.processing_time,
                        message.model_used,
                        json.dumps(message.metadata),
                        json.dumps(message.attachments),
                        message.timestamp.isoformat()
                    ))
                
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store conversation %s: %s", conversation.conversation_id, str(e))
            return False
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Retrieve conversation history."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                
                # Get conversation metadata
                cursor = await db.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                conv_row = await cursor.fetchone()
                
                if not conv_row:
                    return None
                
                # Get messages
                cursor = await db.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                    (conversation_id,)
                )
                message_rows = await cursor.fetchall()
                
                # Build conversation object
                conversation = self._row_to_conversation(conv_row)
                conversation.messages = [self._row_to_message(row) for row in message_rows]
                
                return conversation
        except Exception as e:
            logger.error("Failed to get conversation %s: %s", conversation_id, str(e))
            return None
    
    async def store_user_preferences(self, preferences: UserPreferences) -> bool:
        """Store user preferences."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO user_preferences (
                        user_id, preferred_language, communication_style, detail_level,
                        theme, timezone, date_format, preferred_agents,
                        agent_personalities, data_sharing, analytics_opt_in,
                        personalization, custom_settings, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    preferences.user_id,
                    preferences.preferred_language,
                    preferences.communication_style,
                    preferences.detail_level,
                    preferences.theme,
                    preferences.timezone,
                    preferences.date_format,
                    json.dumps(preferences.preferred_agents),
                    json.dumps(preferences.agent_personalities),
                    preferences.data_sharing,
                    preferences.analytics_opt_in,
                    preferences.personalization,
                    json.dumps(preferences.custom_settings),
                    preferences.created_at.isoformat(),
                    preferences.updated_at.isoformat()
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store user preferences %s: %s", preferences.user_id, str(e))
            return False
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Retrieve user preferences."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_user_preferences(row)
                return None
        except Exception as e:
            logger.error("Failed to get user preferences %s: %s", user_id, str(e))
            return None
    
    async def store_shared_context(self, context: SharedContext) -> bool:
        """Store shared context."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO shared_contexts (
                        context_id, name, scope, data, metadata, owner_id,
                        allowed_agents, allowed_users, version, revision_history,
                        created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.context_id,
                    context.name,
                    context.scope.value,
                    json.dumps(context.data),
                    json.dumps(context.metadata),
                    context.owner_id,
                    json.dumps(context.allowed_agents),
                    json.dumps(context.allowed_users),
                    context.version,
                    json.dumps(context.revision_history),
                    context.created_at.isoformat(),
                    context.updated_at.isoformat(),
                    context.expires_at.isoformat() if context.expires_at else None
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store shared context %s: %s", context.context_id, str(e))
            return False
    
    async def get_shared_context(self, context_id: str) -> Optional[SharedContext]:
        """Retrieve shared context."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM shared_contexts WHERE context_id = ?",
                    (context_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_shared_context(row)
                return None
        except Exception as e:
            logger.error("Failed to get shared context %s: %s", context_id, str(e))
            return None
    
    async def cleanup_expired(self) -> int:
        """Clean up expired data."""
        try:
            now = datetime.now().isoformat()
            deleted_count = 0
            
            async with aiosqlite.connect(str(self.db_path)) as db:
                # Clean up expired sessions
                cursor = await db.execute(
                    "DELETE FROM sessions WHERE expires_at < ?",
                    (now,)
                )
                deleted_count += cursor.rowcount
                
                # Clean up expired shared contexts
                cursor = await db.execute(
                    "DELETE FROM shared_contexts WHERE expires_at < ?",
                    (now,)
                )
                deleted_count += cursor.rowcount
                
                # Clean up old messages based on retention policy
                cutoff = (datetime.now() - timedelta(days=self.config.message_retention_days)).isoformat()
                cursor = await db.execute(
                    "DELETE FROM messages WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted_count += cursor.rowcount
                
                await db.commit()
            
            return deleted_count
        except Exception as e:
            logger.error("Failed to cleanup expired data: %s", str(e))
            return 0
    
    # Helper methods for converting between rows and objects
    def _row_to_session(self, row) -> SessionState:
        """Convert database row to SessionState object."""
        return SessionState(
            session_id=row['session_id'],
            user_id=row['user_id'],
            status=SessionStatus(row['status']),
            session_type=row['session_type'],
            state_data=json.loads(row['state_data']) if row['state_data'] else {},
            variables=json.loads(row['variables']) if row['variables'] else {},
            flags=json.loads(row['flags']) if row['flags'] else {},
            current_conversation_id=row['current_conversation_id'],
            active_agents=json.loads(row['active_agents']) if row['active_agents'] else [],
            current_workflow_id=row['current_workflow_id'],
            total_conversations=row['total_conversations'],
            total_messages=row['total_messages'],
            total_duration=row['total_duration'],
            created_at=datetime.fromisoformat(row['created_at']),
            last_activity=datetime.fromisoformat(row['last_activity']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
        )
    
    def _row_to_conversation(self, row) -> ConversationHistory:
        """Convert database row to ConversationHistory object."""
        return ConversationHistory(
            conversation_id=row['conversation_id'],
            session_id=row['session_id'],
            user_id=row['user_id'],
            title=row['title'],
            summary=row['summary'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            agent_ids=json.loads(row['agent_ids']) if row['agent_ids'] else [],
            message_count=row['message_count'],
            total_tokens=row['total_tokens'],
            user_messages=row['user_messages'],
            assistant_messages=row['assistant_messages'],
            started_at=datetime.fromisoformat(row['started_at']),
            last_activity=datetime.fromisoformat(row['last_activity']),
            ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None
        )
    
    def _row_to_message(self, row) -> Message:
        """Convert database row to Message object."""
        return Message(
            message_id=row['message_id'],
            conversation_id=row['conversation_id'],
            session_id=row['session_id'],
            role=MessageRole(row['role']),
            content=row['content'],
            agent_id=row['agent_id'],
            tool_calls=json.loads(row['tool_calls']) if row['tool_calls'] else [],
            function_call=json.loads(row['function_call']) if row['function_call'] else None,
            thread_id=row['thread_id'],
            tokens_used=row['tokens_used'],
            processing_time=row['processing_time'],
            model_used=row['model_used'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            attachments=json.loads(row['attachments']) if row['attachments'] else [],
            timestamp=datetime.fromisoformat(row['timestamp'])
        )
    
    def _row_to_user_preferences(self, row) -> UserPreferences:
        """Convert database row to UserPreferences object."""
        return UserPreferences(
            user_id=row['user_id'],
            preferred_language=row['preferred_language'],
            communication_style=row['communication_style'],
            detail_level=row['detail_level'],
            theme=row['theme'],
            timezone=row['timezone'],
            date_format=row['date_format'],
            preferred_agents=json.loads(row['preferred_agents']) if row['preferred_agents'] else [],
            agent_personalities=json.loads(row['agent_personalities']) if row['agent_personalities'] else {},
            data_sharing=bool(row['data_sharing']),
            analytics_opt_in=bool(row['analytics_opt_in']),
            personalization=bool(row['personalization']),
            custom_settings=json.loads(row['custom_settings']) if row['custom_settings'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def _row_to_shared_context(self, row) -> SharedContext:
        """Convert database row to SharedContext object."""
        return SharedContext(
            context_id=row['context_id'],
            name=row['name'],
            scope=ContextScope(row['scope']),
            data=json.loads(row['data']) if row['data'] else {},
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            owner_id=row['owner_id'],
            allowed_agents=json.loads(row['allowed_agents']) if row['allowed_agents'] else [],
            allowed_users=json.loads(row['allowed_users']) if row['allowed_users'] else [],
            version=row['version'],
            revision_history=json.loads(row['revision_history']) if row['revision_history'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
        )
    
    # Additional query methods
    async def get_conversations_by_session(self, session_id: str) -> List[ConversationHistory]:
        """Get all conversations for a session."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM conversations WHERE session_id = ? ORDER BY started_at",
                    (session_id,)
                )
                rows = await cursor.fetchall()
                return [self._row_to_conversation(row) for row in rows]
        except Exception as e:
            logger.error("Failed to get conversations for session %s: %s", session_id, str(e))
            return []
    
    async def get_user_sessions(self, user_id: str) -> List[SessionState]:
        """Get all sessions for a user."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,)
                )
                rows = await cursor.fetchall()
                return [self._row_to_session(row) for row in rows]
        except Exception as e:
            logger.error("Failed to get sessions for user %s: %s", user_id, str(e))
            return []
    
    async def search_conversations(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationHistory]:
        """Search conversations."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                
                sql = "SELECT * FROM conversations"
                params = []
                conditions = []
                
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                
                if query:
                    conditions.append("(title LIKE ? OR summary LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY last_activity DESC LIMIT ?"
                params.append(limit)
                
                cursor = await db.execute(sql, params)
                rows = await cursor.fetchall()
                return [self._row_to_conversation(row) for row in rows]
        except Exception as e:
            logger.error("Failed to search conversations: %s", str(e))
            return []
    
    async def get_shared_contexts(self, scope: Optional[ContextScope] = None) -> List[SharedContext]:
        """Get shared contexts by scope."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                
                if scope:
                    cursor = await db.execute(
                        "SELECT * FROM shared_contexts WHERE scope = ? ORDER BY updated_at DESC",
                        (scope.value,)
                    )
                else:
                    cursor = await db.execute(
                        "SELECT * FROM shared_contexts ORDER BY updated_at DESC"
                    )
                
                rows = await cursor.fetchall()
                return [self._row_to_shared_context(row) for row in rows]
        except Exception as e:
            logger.error("Failed to get shared contexts: %s", str(e))
            return []
    
    async def store_agent_context(self, context: AgentContext) -> bool:
        """Store agent context."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO agent_contexts (
                        agent_id, session_id, current_task, task_history, goals,
                        working_memory, facts, assumptions, available_tools,
                        tool_usage_history, performance_metrics, error_history,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.agent_id,
                    context.session_id,
                    context.current_task,
                    json.dumps(context.task_history),
                    json.dumps(context.goals),
                    json.dumps(context.working_memory),
                    json.dumps(context.facts),
                    json.dumps(context.assumptions),
                    json.dumps(context.available_tools),
                    json.dumps(context.tool_usage_history),
                    json.dumps(context.performance_metrics),
                    json.dumps(context.error_history),
                    context.created_at.isoformat(),
                    context.updated_at.isoformat()
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store agent context %s:%s: %s", 
                        context.agent_id, context.session_id, str(e))
            return False
    
    async def get_agent_context(self, agent_id: str, session_id: str) -> Optional[AgentContext]:
        """Get agent context."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM agent_contexts WHERE agent_id = ? AND session_id = ?",
                    (agent_id, session_id)
                )
                row = await cursor.fetchone()
                
                if row:
                    return AgentContext(
                        agent_id=row['agent_id'],
                        session_id=row['session_id'],
                        current_task=row['current_task'],
                        task_history=json.loads(row['task_history']) if row['task_history'] else [],
                        goals=json.loads(row['goals']) if row['goals'] else [],
                        working_memory=json.loads(row['working_memory']) if row['working_memory'] else {},
                        facts=json.loads(row['facts']) if row['facts'] else [],
                        assumptions=json.loads(row['assumptions']) if row['assumptions'] else [],
                        available_tools=json.loads(row['available_tools']) if row['available_tools'] else [],
                        tool_usage_history=json.loads(row['tool_usage_history']) if row['tool_usage_history'] else [],
                        performance_metrics=json.loads(row['performance_metrics']) if row['performance_metrics'] else {},
                        error_history=json.loads(row['error_history']) if row['error_history'] else [],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                return None
        except Exception as e:
            logger.error("Failed to get agent context %s:%s: %s", agent_id, session_id, str(e))
            return None
    
    async def delete_shared_context(self, context_id: str) -> bool:
        """Delete shared context."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("DELETE FROM shared_contexts WHERE context_id = ?", (context_id,))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete shared context %s: %s", context_id, str(e))
            return False
    
    # Count methods for analytics
    async def count_sessions(self) -> int:
        """Count total sessions."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM sessions")
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error("Failed to count sessions: %s", str(e))
            return 0
    
    async def count_conversations(self) -> int:
        """Count total conversations."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM conversations")
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error("Failed to count conversations: %s", str(e))
            return 0
    
    async def count_messages(self) -> int:
        """Count total messages."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM messages")
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error("Failed to count messages: %s", str(e))
            return 0
    
    async def count_shared_contexts(self) -> int:
        """Count shared contexts."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM shared_contexts")
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error("Failed to count shared contexts: %s", str(e))
            return 0


class InMemoryContextStore(BaseContextStore):
    """In-memory context storage for testing."""
    
    def __init__(self, config: ContextConfig):
        super().__init__(config)
        self.sessions: Dict[str, SessionState] = {}
        self.conversations: Dict[str, ConversationHistory] = {}
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.shared_contexts: Dict[str, SharedContext] = {}
        self.agent_contexts: Dict[str, AgentContext] = {}
    
    async def store_session(self, session: SessionState) -> bool:
        self.sessions[session.session_id] = session
        return True
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)
    
    async def store_conversation(self, conversation: ConversationHistory) -> bool:
        self.conversations[conversation.conversation_id] = conversation
        return True
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        return self.conversations.get(conversation_id)
    
    async def store_user_preferences(self, preferences: UserPreferences) -> bool:
        self.user_preferences[preferences.user_id] = preferences
        return True
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        return self.user_preferences.get(user_id)
    
    async def store_shared_context(self, context: SharedContext) -> bool:
        self.shared_contexts[context.context_id] = context
        return True
    
    async def get_shared_context(self, context_id: str) -> Optional[SharedContext]:
        return self.shared_contexts.get(context_id)
    
    async def cleanup_expired(self) -> int:
        # Simple cleanup for in-memory store
        count = 0
        now = datetime.now()
        
        # Clean sessions
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session.expires_at and session.expires_at < now
        ]
        for sid in expired_sessions:
            del self.sessions[sid]
            count += 1
        
        # Clean shared contexts
        expired_contexts = [
            cid for cid, context in self.shared_contexts.items()
            if context.expires_at and context.expires_at < now
        ]
        for cid in expired_contexts:
            del self.shared_contexts[cid]
            count += 1
        
        return count


class ContextStoreManager:
    """Manager for context storage operations."""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        
        # Initialize store based on configuration
        if config.storage_path and config.storage_path != ":memory:":
            self.store = SQLiteContextStore(config)
        else:
            self.store = InMemoryContextStore(config)
    
    async def initialize(self):
        """Initialize the store manager."""
        await self.store.initialize()
    
    # Delegate all operations to the underlying store
    async def store_session(self, session: SessionState) -> bool:
        return await self.store.store_session(session)
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        return await self.store.get_session(session_id)
    
    async def store_conversation(self, conversation: ConversationHistory) -> bool:
        return await self.store.store_conversation(conversation)
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        return await self.store.get_conversation(conversation_id)
    
    async def store_user_preferences(self, preferences: UserPreferences) -> bool:
        return await self.store.store_user_preferences(preferences)
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        return await self.store.get_user_preferences(user_id)
    
    async def store_shared_context(self, context: SharedContext) -> bool:
        return await self.store.store_shared_context(context)
    
    async def get_shared_context(self, context_id: str) -> Optional[SharedContext]:
        return await self.store.get_shared_context(context_id)
    
    async def cleanup_expired(self) -> int:
        return await self.store.cleanup_expired()
    
    # Additional methods specific to SQLite store
    async def get_conversations_by_session(self, session_id: str) -> List[ConversationHistory]:
        if hasattr(self.store, 'get_conversations_by_session'):
            return await self.store.get_conversations_by_session(session_id)
        return []
    
    async def get_user_sessions(self, user_id: str) -> List[SessionState]:
        if hasattr(self.store, 'get_user_sessions'):
            return await self.store.get_user_sessions(user_id)
        return []
    
    async def search_conversations(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationHistory]:
        if hasattr(self.store, 'search_conversations'):
            return await self.store.search_conversations(user_id, query, limit)
        return []
    
    async def get_shared_contexts(self, scope: Optional[ContextScope] = None) -> List[SharedContext]:
        if hasattr(self.store, 'get_shared_contexts'):
            return await self.store.get_shared_contexts(scope)
        return []
    
    async def store_agent_context(self, context: AgentContext) -> bool:
        if hasattr(self.store, 'store_agent_context'):
            return await self.store.store_agent_context(context)
        return False
    
    async def get_agent_context(self, agent_id: str, session_id: str) -> Optional[AgentContext]:
        if hasattr(self.store, 'get_agent_context'):
            return await self.store.get_agent_context(agent_id, session_id)
        return None
    
    async def delete_shared_context(self, context_id: str) -> bool:
        if hasattr(self.store, 'delete_shared_context'):
            return await self.store.delete_shared_context(context_id)
        return False
    
    # Count methods for analytics
    async def count_sessions(self) -> int:
        if hasattr(self.store, 'count_sessions'):
            return await self.store.count_sessions()
        return len(getattr(self.store, 'sessions', {}))
    
    async def count_conversations(self) -> int:
        if hasattr(self.store, 'count_conversations'):
            return await self.store.count_conversations()
        return len(getattr(self.store, 'conversations', {}))
    
    async def count_messages(self) -> int:
        if hasattr(self.store, 'count_messages'):
            return await self.store.count_messages()
        # For in-memory store, count messages across all conversations
        total = 0
        for conv in getattr(self.store, 'conversations', {}).values():
            total += len(conv.messages)
        return total
    
    async def count_shared_contexts(self) -> int:
        if hasattr(self.store, 'count_shared_contexts'):
            return await self.store.count_shared_contexts()
        return len(getattr(self.store, 'shared_contexts', {}))
    
    async def shutdown(self):
        """Shutdown the store manager."""
        if hasattr(self.store, 'shutdown'):
            await self.store.shutdown()
        logger.info("ContextStoreManager shutdown complete")
