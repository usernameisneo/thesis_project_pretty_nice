"""
Writing Session Management for the AI-Powered Thesis Assistant.

This module implements comprehensive writing session tracking with
productivity analytics, goal setting, and AI-powered assistance.

Features:
    - Session tracking and timing
    - Productivity analytics
    - Goal setting and achievement
    - Writing habit analysis
    - AI-powered writing assistance
    - Progress visualization

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Writing session status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SessionType(Enum):
    """Writing session type enumeration."""
    WRITING = "writing"
    EDITING = "editing"
    RESEARCH = "research"
    PLANNING = "planning"
    REVIEW = "review"


@dataclass
class SessionGoal:
    """Writing session goal structure."""
    type: str  # words, pages, time
    target: int
    achieved: int = 0
    completed: bool = False


@dataclass
class SessionMetrics:
    """Writing session metrics."""
    words_written: int = 0
    words_deleted: int = 0
    net_words: int = 0
    characters_typed: int = 0
    backspaces: int = 0
    typing_speed: float = 0.0  # words per minute
    focus_time: float = 0.0  # actual writing time
    break_time: float = 0.0  # pause time
    productivity_score: float = 0.0


@dataclass
class WritingSessionData:
    """Writing session data structure."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0  # minutes
    status: SessionStatus = SessionStatus.ACTIVE
    session_type: SessionType = SessionType.WRITING
    chapter_path: Optional[str] = None
    project_path: Optional[str] = None
    goals: List[SessionGoal] = field(default_factory=list)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    notes: str = ""
    tags: List[str] = field(default_factory=list)


class WritingSession:
    """
    Comprehensive writing session management system.
    
    This class provides complete session tracking with productivity
    analytics, goal management, and AI-powered assistance.
    """
    
    def __init__(self, session_data_dir: str):
        """
        Initialize writing session manager.
        
        Args:
            session_data_dir: Directory for session data storage
        """
        self.session_data_dir = Path(session_data_dir)
        self.session_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[WritingSessionData] = None
        self.session_start_time: Optional[float] = None
        self.last_activity_time: Optional[float] = None
        self.pause_start_time: Optional[float] = None
        
        # Session tracking
        self.initial_content: str = ""
        self.keystroke_count: int = 0
        self.backspace_count: int = 0
        
        logger.info("Writing session manager initialized")
    
    def start_session(self, session_type: SessionType = SessionType.WRITING,
                     chapter_path: Optional[str] = None,
                     project_path: Optional[str] = None,
                     goals: Optional[List[SessionGoal]] = None) -> str:
        """
        Start a new writing session.
        
        Args:
            session_type: Type of writing session
            chapter_path: Path to chapter being worked on
            project_path: Path to project
            goals: Session goals
            
        Returns:
            Session ID
        """
        try:
            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create session data
            self.current_session = WritingSessionData(
                session_id=session_id,
                start_time=datetime.now(),
                session_type=session_type,
                chapter_path=chapter_path,
                project_path=project_path,
                goals=goals or []
            )
            
            # Initialize tracking
            self.session_start_time = time.time()
            self.last_activity_time = self.session_start_time
            self.keystroke_count = 0
            self.backspace_count = 0
            
            # Load initial content for comparison
            if chapter_path and Path(chapter_path).exists():
                self.initial_content = Path(chapter_path).read_text(encoding='utf-8')
            else:
                self.initial_content = ""
            
            logger.info(f"Writing session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise Exception(f"Session start failed: {e}")
    
    def pause_session(self) -> None:
        """Pause the current writing session."""
        if not self.current_session or self.current_session.status != SessionStatus.ACTIVE:
            raise Exception("No active session to pause")
        
        self.current_session.status = SessionStatus.PAUSED
        self.pause_start_time = time.time()
        
        logger.info(f"Session paused: {self.current_session.session_id}")
    
    def resume_session(self) -> None:
        """Resume a paused writing session."""
        if not self.current_session or self.current_session.status != SessionStatus.PAUSED:
            raise Exception("No paused session to resume")
        
        # Calculate break time
        if self.pause_start_time:
            break_duration = time.time() - self.pause_start_time
            self.current_session.metrics.break_time += break_duration / 60  # convert to minutes
        
        self.current_session.status = SessionStatus.ACTIVE
        self.last_activity_time = time.time()
        self.pause_start_time = None
        
        logger.info(f"Session resumed: {self.current_session.session_id}")
    
    def end_session(self, notes: str = "") -> WritingSessionData:
        """
        End the current writing session.
        
        Args:
            notes: Session notes
            
        Returns:
            Completed session data
        """
        if not self.current_session:
            raise Exception("No active session to end")
        
        try:
            # Update session data
            self.current_session.end_time = datetime.now()
            self.current_session.status = SessionStatus.COMPLETED
            self.current_session.notes = notes
            
            # Calculate duration
            if self.session_start_time:
                total_duration = time.time() - self.session_start_time
                self.current_session.duration = total_duration / 60  # convert to minutes
            
            # Calculate metrics
            self._calculate_session_metrics()
            
            # Check goal achievement
            self._check_goal_achievement()
            
            # Save session data
            self._save_session_data()
            
            logger.info(f"Session completed: {self.current_session.session_id}")
            
            completed_session = self.current_session
            self.current_session = None
            
            return completed_session
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            raise Exception(f"Session end failed: {e}")
    
    def cancel_session(self) -> None:
        """Cancel the current writing session."""
        if not self.current_session:
            raise Exception("No active session to cancel")
        
        self.current_session.status = SessionStatus.CANCELLED
        self.current_session.end_time = datetime.now()
        
        # Save cancelled session data
        self._save_session_data()
        
        logger.info(f"Session cancelled: {self.current_session.session_id}")
        self.current_session = None
    
    def update_activity(self, keystroke: bool = False, backspace: bool = False) -> None:
        """
        Update session activity tracking.
        
        Args:
            keystroke: Whether a keystroke occurred
            backspace: Whether a backspace occurred
        """
        if not self.current_session or self.current_session.status != SessionStatus.ACTIVE:
            return
        
        current_time = time.time()
        self.last_activity_time = current_time
        
        if keystroke:
            self.keystroke_count += 1
            self.current_session.metrics.characters_typed += 1
        
        if backspace:
            self.backspace_count += 1
            self.current_session.metrics.backspaces += 1
    
    def _calculate_session_metrics(self) -> None:
        """Calculate comprehensive session metrics."""
        if not self.current_session:
            return
        
        # Calculate focus time (total time minus breaks)
        total_time = self.current_session.duration
        break_time = self.current_session.metrics.break_time
        self.current_session.metrics.focus_time = max(0, total_time - break_time)
        
        # Calculate word metrics if chapter content available
        if self.current_session.chapter_path and Path(self.current_session.chapter_path).exists():
            current_content = Path(self.current_session.chapter_path).read_text(encoding='utf-8')
            
            # Count words
            initial_words = len(self.initial_content.split())
            current_words = len(current_content.split())
            
            # Calculate net word change
            self.current_session.metrics.net_words = current_words - initial_words
            
            # Estimate words written and deleted (simplified)
            if self.current_session.metrics.net_words >= 0:
                self.current_session.metrics.words_written = self.current_session.metrics.net_words
                self.current_session.metrics.words_deleted = 0
            else:
                self.current_session.metrics.words_written = 0
                self.current_session.metrics.words_deleted = abs(self.current_session.metrics.net_words)
        
        # Calculate typing speed (words per minute)
        if self.current_session.metrics.focus_time > 0:
            self.current_session.metrics.typing_speed = (
                self.current_session.metrics.words_written / self.current_session.metrics.focus_time
            )
        
        # Calculate productivity score (0-100)
        self.current_session.metrics.productivity_score = self._calculate_productivity_score()
    
    def _calculate_productivity_score(self) -> float:
        """Calculate productivity score based on various factors."""
        if not self.current_session:
            return 0.0
        
        score = 0.0
        metrics = self.current_session.metrics
        
        # Words written factor (40% of score)
        if metrics.words_written > 0:
            # Normalize to 0-40 based on reasonable writing speed
            words_score = min(40, (metrics.words_written / 500) * 40)
            score += words_score
        
        # Focus time factor (30% of score)
        if self.current_session.duration > 0:
            focus_ratio = metrics.focus_time / self.current_session.duration
            focus_score = focus_ratio * 30
            score += focus_score
        
        # Typing efficiency factor (20% of score)
        if metrics.characters_typed > 0:
            efficiency_ratio = 1 - (metrics.backspaces / metrics.characters_typed)
            efficiency_score = max(0, efficiency_ratio) * 20
            score += efficiency_score
        
        # Goal achievement factor (10% of score)
        if self.current_session.goals:
            achieved_goals = sum(1 for goal in self.current_session.goals if goal.completed)
            goal_ratio = achieved_goals / len(self.current_session.goals)
            goal_score = goal_ratio * 10
            score += goal_score
        
        return min(100.0, score)
    
    def _check_goal_achievement(self) -> None:
        """Check and update goal achievement status."""
        if not self.current_session:
            return
        
        for goal in self.current_session.goals:
            if goal.type == "words":
                goal.achieved = self.current_session.metrics.words_written
            elif goal.type == "time":
                goal.achieved = int(self.current_session.metrics.focus_time)
            elif goal.type == "pages":
                # Estimate pages (250 words per page)
                goal.achieved = max(1, self.current_session.metrics.words_written // 250)
            
            # Check if goal is completed
            goal.completed = goal.achieved >= goal.target
    
    def _save_session_data(self) -> None:
        """Save session data to file."""
        if not self.current_session:
            return
        
        try:
            session_file = self.session_data_dir / f"{self.current_session.session_id}.json"
            
            # Convert to dictionary
            data = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'duration': self.current_session.duration,
                'status': self.current_session.status.value,
                'session_type': self.current_session.session_type.value,
                'chapter_path': self.current_session.chapter_path,
                'project_path': self.current_session.project_path,
                'goals': [
                    {
                        'type': goal.type,
                        'target': goal.target,
                        'achieved': goal.achieved,
                        'completed': goal.completed
                    }
                    for goal in self.current_session.goals
                ],
                'metrics': {
                    'words_written': self.current_session.metrics.words_written,
                    'words_deleted': self.current_session.metrics.words_deleted,
                    'net_words': self.current_session.metrics.net_words,
                    'characters_typed': self.current_session.metrics.characters_typed,
                    'backspaces': self.current_session.metrics.backspaces,
                    'typing_speed': self.current_session.metrics.typing_speed,
                    'focus_time': self.current_session.metrics.focus_time,
                    'break_time': self.current_session.metrics.break_time,
                    'productivity_score': self.current_session.metrics.productivity_score
                },
                'notes': self.current_session.notes,
                'tags': self.current_session.tags
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Session data saved: {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def get_session_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get writing session history.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of session data
        """
        try:
            sessions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Load all session files
            for session_file in self.session_data_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Check if session is within date range
                    session_date = datetime.fromisoformat(session_data['start_time'])
                    if session_date >= cutoff_date:
                        sessions.append(session_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to load session file {session_file}: {e}")
            
            # Sort by start time (newest first)
            sessions.sort(key=lambda x: x['start_time'], reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    def get_productivity_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get productivity analytics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        sessions = self.get_session_history(days)
        
        if not sessions:
            return {}
        
        # Calculate analytics
        total_sessions = len(sessions)
        total_time = sum(session.get('duration', 0) for session in sessions)
        total_words = sum(session.get('metrics', {}).get('words_written', 0) for session in sessions)
        avg_productivity = sum(session.get('metrics', {}).get('productivity_score', 0) for session in sessions) / total_sessions
        
        # Daily breakdown
        daily_stats = {}
        for session in sessions:
            date = session['start_time'][:10]  # YYYY-MM-DD
            if date not in daily_stats:
                daily_stats[date] = {'sessions': 0, 'time': 0, 'words': 0}
            
            daily_stats[date]['sessions'] += 1
            daily_stats[date]['time'] += session.get('duration', 0)
            daily_stats[date]['words'] += session.get('metrics', {}).get('words_written', 0)
        
        return {
            'period_days': days,
            'total_sessions': total_sessions,
            'total_time_minutes': total_time,
            'total_words_written': total_words,
            'average_productivity_score': avg_productivity,
            'average_session_length': total_time / total_sessions if total_sessions > 0 else 0,
            'average_words_per_session': total_words / total_sessions if total_sessions > 0 else 0,
            'daily_breakdown': daily_stats
        }
