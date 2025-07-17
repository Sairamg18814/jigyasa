"""
Real Continuous Learning System using Llama 3.2
Implements actual learning from interactions and feedback
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import logging

from ..models.ollama_wrapper import OllamaWrapper

class RealContinuousLearner:
    """Implements actual continuous learning with memory and adaptation"""
    
    def __init__(self, ollama_wrapper: Optional[OllamaWrapper] = None):
        self.ollama = ollama_wrapper or OllamaWrapper()
        self.logger = logging.getLogger(__name__)
        
        # Initialize knowledge database
        self.db_path = Path(".jigyasa_knowledge/learning.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
        
        # Learning metrics
        self.performance_history = []
        self.adaptation_strategies = {}
        
    def init_database(self):
        """Initialize SQLite database for knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Knowledge table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                context TEXT,
                learned_insight TEXT,
                confidence REAL,
                times_used INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """)
        
        # Interaction history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_text TEXT,
                output_text TEXT,
                feedback TEXT,
                success BOOLEAN,
                knowledge_ids TEXT
            )
        """)
        
        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def learn_from_interaction(self, input_text: str, output_text: str, 
                             feedback: Optional[str] = None, success: bool = True) -> Dict[str, Any]:
        """Learn from a single interaction"""
        
        # Extract insights using Llama 3.2
        prompt = f"""Analyze this interaction and extract key learnings:

Input: {input_text}
Output: {output_text}
Feedback: {feedback or 'No explicit feedback'}
Success: {success}

Extract:
1. Key insights that should be remembered
2. Patterns to recognize in future
3. Improvements for similar situations
4. Confidence level (0-1) in the learning

Format as JSON."""

        response = self.ollama.generate(prompt, temperature=0.2)
        
        # Parse learning insights
        try:
            insights = self._parse_json_from_response(response.text)
        except:
            insights = {
                "key_insights": ["Interaction processed"],
                "patterns": [],
                "improvements": [],
                "confidence": 0.5
            }
            
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        knowledge_ids = []
        
        # Store each insight
        for insight in insights.get("key_insights", []):
            cursor.execute("""
                INSERT INTO knowledge (timestamp, context, learned_insight, confidence)
                VALUES (?, ?, ?, ?)
            """, (timestamp, input_text, insight, insights.get("confidence", 0.5)))
            knowledge_ids.append(cursor.lastrowid)
            
        # Store interaction
        cursor.execute("""
            INSERT INTO interactions (timestamp, input_text, output_text, feedback, success, knowledge_ids)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, input_text, output_text, feedback, success, json.dumps(knowledge_ids)))
        
        conn.commit()
        conn.close()
        
        # Update performance metrics
        self._update_metrics(success, insights.get("confidence", 0.5))
        
        return {
            "insights_learned": len(insights.get("key_insights", [])),
            "confidence": insights.get("confidence", 0.5),
            "patterns_identified": len(insights.get("patterns", [])),
            "knowledge_ids": knowledge_ids
        }
        
    def apply_learned_knowledge(self, context: str) -> Tuple[str, List[Dict]]:
        """Apply previously learned knowledge to new context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find relevant knowledge
        cursor.execute("""
            SELECT learned_insight, confidence, times_used, success_rate
            FROM knowledge
            WHERE context LIKE ?
            ORDER BY confidence * success_rate DESC
            LIMIT 10
        """, (f"%{context[:50]}%",))
        
        relevant_knowledge = cursor.fetchall()
        conn.close()
        
        if not relevant_knowledge:
            return "No relevant knowledge found", []
            
        # Build knowledge-enhanced prompt
        knowledge_context = "\n".join([
            f"- {insight} (confidence: {conf:.2f}, success: {succ:.2f})"
            for insight, conf, used, succ in relevant_knowledge
        ])
        
        prompt = f"""Using the following learned knowledge:

{knowledge_context}

Respond to: {context}

Apply the most relevant insights and patterns."""

        response = self.ollama.generate(prompt, temperature=0.3)
        
        # Update usage statistics
        self._update_knowledge_usage([k[0] for k in relevant_knowledge])
        
        return response.text, [
            {
                "insight": k[0],
                "confidence": k[1],
                "times_used": k[2],
                "success_rate": k[3]
            }
            for k in relevant_knowledge
        ]
        
    def train_on_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Train on a dataset to build knowledge base"""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                
            total_learned = 0
            total_patterns = 0
            
            for item in data:
                if 'input' in item and 'output' in item:
                    result = self.learn_from_interaction(
                        item['input'],
                        item['output'],
                        item.get('feedback'),
                        item.get('success', True)
                    )
                    total_learned += result['insights_learned']
                    total_patterns += result['patterns_identified']
                    
            return {
                "status": "success",
                "total_items": len(data),
                "insights_learned": total_learned,
                "patterns_identified": total_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total knowledge
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        total_knowledge = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM knowledge")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Success rate
        cursor.execute("SELECT AVG(success) FROM interactions")
        success_rate = cursor.fetchone()[0] or 0
        
        # Most used knowledge
        cursor.execute("""
            SELECT learned_insight, times_used, success_rate
            FROM knowledge
            ORDER BY times_used DESC
            LIMIT 5
        """)
        top_knowledge = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_knowledge_items": total_knowledge,
            "average_confidence": avg_confidence,
            "overall_success_rate": success_rate,
            "top_applied_knowledge": [
                {
                    "insight": k[0],
                    "times_used": k[1],
                    "success_rate": k[2]
                }
                for k in top_knowledge
            ],
            "learning_velocity": self._calculate_learning_velocity()
        }
        
    def _parse_json_from_response(self, response: str) -> Dict:
        """Extract JSON from LLM response"""
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
        
    def _update_metrics(self, success: bool, confidence: float):
        """Update performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO metrics (timestamp, metric_name, metric_value, context)
            VALUES (?, ?, ?, ?)
        """, (timestamp, "interaction_success", 1.0 if success else 0.0, ""))
        
        cursor.execute("""
            INSERT INTO metrics (timestamp, metric_name, metric_value, context)
            VALUES (?, ?, ?, ?)
        """, (timestamp, "learning_confidence", confidence, ""))
        
        conn.commit()
        conn.close()
        
    def _update_knowledge_usage(self, insights: List[str]):
        """Update usage statistics for applied knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for insight in insights:
            cursor.execute("""
                UPDATE knowledge
                SET times_used = times_used + 1
                WHERE learned_insight = ?
            """, (insight,))
            
        conn.commit()
        conn.close()
        
    def _calculate_learning_velocity(self) -> float:
        """Calculate rate of learning over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get knowledge acquisition over last 7 days
        cursor.execute("""
            SELECT DATE(timestamp), COUNT(*)
            FROM knowledge
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
        """)
        
        daily_learning = cursor.fetchall()
        conn.close()
        
        if len(daily_learning) > 1:
            counts = [d[1] for d in daily_learning]
            # Simple linear regression slope
            x = np.arange(len(counts))
            slope = np.polyfit(x, counts, 1)[0]
            return float(slope)
            
        return 0.0