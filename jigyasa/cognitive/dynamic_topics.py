"""
Dynamic Topic Discovery for SEAL Training
Automatically discovers trending and relevant topics for continuous learning
"""

import random
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import logging
from bs4 import BeautifulSoup


class DynamicTopicGenerator:
    """
    Generates dynamic topics for continuous learning based on:
    - Current trends
    - Diverse knowledge domains
    - User interactions
    - Model performance gaps
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Base knowledge domains for balanced learning
        self.knowledge_domains = {
            "science": [
                "quantum computing", "machine learning", "neuroscience", 
                "climate science", "astronomy", "biology", "chemistry",
                "physics", "mathematics", "computer science"
            ],
            "technology": [
                "artificial intelligence", "blockchain", "cybersecurity",
                "robotics", "biotechnology", "nanotechnology", "5G",
                "renewable energy", "space technology", "quantum technology"
            ],
            "humanities": [
                "philosophy", "history", "literature", "psychology",
                "sociology", "anthropology", "linguistics", "art",
                "music", "cultural studies"
            ],
            "current_affairs": [
                "global economics", "environmental issues", "public health",
                "international relations", "social justice", "innovation",
                "education", "future of work", "digital transformation"
            ],
            "specialized": [
                "cognitive science", "systems thinking", "complex systems",
                "emergence", "consciousness studies", "ethics in AI",
                "decision theory", "game theory", "information theory"
            ]
        }
        
        # Topic templates for generating specific queries
        self.topic_templates = [
            "latest developments in {topic}",
            "fundamental concepts of {topic}",
            "applications of {topic}",
            "future of {topic}",
            "ethics and {topic}",
            "history of {topic}",
            "{topic} and society",
            "innovations in {topic}",
            "challenges in {topic}",
            "breakthroughs in {topic}"
        ]
        
        # Track recently used topics to ensure diversity
        self.recent_topics = []
        self.max_recent = 50
    
    def generate_dynamic_topics(
        self, 
        num_topics: int = 10,
        include_trends: bool = True,
        diversity_weight: float = 0.7
    ) -> List[str]:
        """
        Generate a diverse set of topics for learning
        
        Args:
            num_topics: Number of topics to generate
            include_trends: Whether to try to include trending topics
            diversity_weight: Weight for ensuring topic diversity (0-1)
        
        Returns:
            List of topic strings
        """
        topics = []
        
        # 1. Get some trending topics if available
        if include_trends:
            trending = self._get_trending_topics()
            topics.extend(trending[:num_topics // 3])
        
        # 2. Add topics from different knowledge domains
        remaining_slots = num_topics - len(topics)
        domain_topics = self._get_diverse_domain_topics(remaining_slots, diversity_weight)
        topics.extend(domain_topics)
        
        # 3. Apply topic templates for variety
        enhanced_topics = []
        for topic in topics:
            if random.random() < 0.5:  # 50% chance to use template
                template = random.choice(self.topic_templates)
                enhanced_topic = template.format(topic=topic)
                enhanced_topics.append(enhanced_topic)
            else:
                enhanced_topics.append(topic)
        
        # 4. Ensure uniqueness and update recent topics
        unique_topics = []
        for topic in enhanced_topics:
            if topic not in self.recent_topics and topic not in unique_topics:
                unique_topics.append(topic)
        
        # Update recent topics (FIFO)
        self.recent_topics.extend(unique_topics)
        if len(self.recent_topics) > self.max_recent:
            self.recent_topics = self.recent_topics[-self.max_recent:]
        
        self.logger.info(f"Generated {len(unique_topics)} dynamic topics")
        return unique_topics[:num_topics]
    
    def _get_trending_topics(self) -> List[str]:
        """
        Get trending topics from various sources
        Note: In production, this would use real APIs
        """
        # Simulated trending topics based on current tech trends
        trending_topics = [
            "large language models",
            "sustainable technology",
            "quantum supremacy",
            "metaverse development",
            "gene editing CRISPR",
            "neural interfaces",
            "climate change solutions",
            "space exploration",
            "renewable energy storage",
            "autonomous systems"
        ]
        
        # Add some randomization based on "current" trends
        current_month = datetime.now().month
        seasonal_topics = {
            1: ["new year technology predictions", "emerging tech trends"],
            4: ["spring research breakthroughs", "environmental sustainability"],
            7: ["summer innovation projects", "outdoor technology"],
            10: ["Nobel prize discoveries", "breakthrough science"],
            12: ["year in review technology", "future predictions"]
        }
        
        month_topics = seasonal_topics.get(current_month, [])
        trending_topics.extend(month_topics)
        
        # Shuffle and return subset
        random.shuffle(trending_topics)
        return trending_topics[:5]
    
    def _get_diverse_domain_topics(
        self, 
        num_topics: int, 
        diversity_weight: float
    ) -> List[str]:
        """
        Get topics ensuring diversity across knowledge domains
        """
        topics = []
        domains = list(self.knowledge_domains.keys())
        
        # Calculate topics per domain
        topics_per_domain = max(1, num_topics // len(domains))
        
        for domain in domains:
            domain_topics = self.knowledge_domains[domain]
            
            # Filter out recently used topics
            available_topics = [
                t for t in domain_topics 
                if t not in self.recent_topics
            ]
            
            if not available_topics:
                available_topics = domain_topics
            
            # Select topics from this domain
            num_select = min(topics_per_domain, len(available_topics))
            if diversity_weight > random.random():
                # High diversity: random selection
                selected = random.sample(available_topics, num_select)
            else:
                # Lower diversity: pick first (most important) topics
                selected = available_topics[:num_select]
            
            topics.extend(selected)
        
        random.shuffle(topics)
        return topics[:num_topics]
    
    def get_topic_metadata(self, topic: str) -> Dict[str, any]:
        """
        Get metadata about a topic for better learning
        """
        # Determine domain
        domain = "general"
        for d, topics in self.knowledge_domains.items():
            if any(t in topic.lower() for t in topics):
                domain = d
                break
        
        # Estimate complexity
        complexity_keywords = {
            "high": ["quantum", "advanced", "complex", "theoretical", "mathematical"],
            "medium": ["applications", "development", "systems", "analysis"],
            "low": ["introduction", "basics", "fundamental", "overview"]
        }
        
        complexity = "medium"
        for level, keywords in complexity_keywords.items():
            if any(kw in topic.lower() for kw in keywords):
                complexity = level
                break
        
        return {
            "topic": topic,
            "domain": domain,
            "complexity": complexity,
            "timestamp": datetime.now().isoformat(),
            "learning_priority": self._calculate_priority(topic, domain, complexity)
        }
    
    def _calculate_priority(self, topic: str, domain: str, complexity: str) -> float:
        """
        Calculate learning priority for a topic
        """
        base_priority = 0.5
        
        # Boost priority for certain domains
        domain_weights = {
            "science": 0.9,
            "technology": 0.85,
            "current_affairs": 0.8,
            "specialized": 0.95,
            "humanities": 0.7
        }
        
        priority = base_priority * domain_weights.get(domain, 0.7)
        
        # Adjust for complexity
        if complexity == "high":
            priority *= 1.2
        elif complexity == "low":
            priority *= 0.8
        
        # Boost if topic contains certain keywords
        important_keywords = [
            "AI", "future", "breakthrough", "innovation", 
            "critical", "emerging", "revolutionary"
        ]
        
        if any(kw.lower() in topic.lower() for kw in important_keywords):
            priority *= 1.15
        
        return min(1.0, priority)
    
    def generate_learning_curriculum(
        self, 
        duration_days: int = 30,
        topics_per_day: int = 3
    ) -> List[Dict[str, any]]:
        """
        Generate a learning curriculum for specified duration
        """
        curriculum = []
        total_topics_needed = duration_days * topics_per_day
        
        # Generate topics with high diversity
        all_topics = self.generate_dynamic_topics(
            num_topics=total_topics_needed,
            include_trends=True,
            diversity_weight=0.8
        )
        
        # Organize into daily batches
        for day in range(duration_days):
            start_idx = day * topics_per_day
            end_idx = start_idx + topics_per_day
            
            daily_topics = all_topics[start_idx:end_idx]
            
            # Get metadata for each topic
            daily_plan = {
                "day": day + 1,
                "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                "topics": [self.get_topic_metadata(topic) for topic in daily_topics],
                "focus_area": self._get_daily_focus(day)
            }
            
            curriculum.append(daily_plan)
        
        return curriculum
    
    def _get_daily_focus(self, day: int) -> str:
        """
        Determine daily focus area based on day number
        """
        focus_cycle = [
            "Foundational Knowledge",
            "Current Applications", 
            "Future Directions",
            "Ethical Considerations",
            "Practical Implementation",
            "Theoretical Deep Dive",
            "Cross-Domain Connections"
        ]
        
        return focus_cycle[day % len(focus_cycle)]


# Example usage
if __name__ == "__main__":
    generator = DynamicTopicGenerator()
    
    # Generate some topics
    topics = generator.generate_dynamic_topics(num_topics=10)
    print("\nGenerated Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    # Generate a weekly curriculum
    print("\n\nWeekly Learning Curriculum:")
    curriculum = generator.generate_learning_curriculum(duration_days=7, topics_per_day=3)
    
    for day_plan in curriculum:
        print(f"\nDay {day_plan['day']} - {day_plan['date']} ({day_plan['focus_area']}):")
        for topic_info in day_plan['topics']:
            print(f"  - {topic_info['topic']} [Domain: {topic_info['domain']}, Priority: {topic_info['learning_priority']:.2f}]")