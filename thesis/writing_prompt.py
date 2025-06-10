"""
AI-Powered Writing Prompts for the Thesis Assistant.

This module provides intelligent writing assistance with contextual prompts,
research suggestions, and AI-powered content generation.

Features:
    - Contextual writing prompts
    - Research question generation
    - Content structure suggestions
    - Citation recommendations
    - Writing style analysis
    - Academic language enhancement

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Local imports
from api.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Writing prompt type enumeration."""
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    TRANSITION = "transition"
    ARGUMENT = "argument"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"


class PromptCategory(Enum):
    """Prompt category enumeration."""
    STRUCTURE = "structure"
    CONTENT = "content"
    STYLE = "style"
    RESEARCH = "research"
    CITATION = "citation"
    REVISION = "revision"


@dataclass
class WritingPromptResponse:
    """Writing prompt response structure."""
    prompt_type: PromptType
    category: PromptCategory
    suggestions: List[str]
    research_questions: List[str]
    citation_recommendations: List[str]
    style_improvements: List[str]
    next_steps: List[str]
    confidence_score: float
    generated_at: datetime


class WritingPrompt:
    """
    AI-powered writing prompt and assistance system.
    
    This class provides intelligent writing assistance with contextual
    prompts, research suggestions, and content generation.
    """
    
    def __init__(self, openrouter_client: Optional[OpenRouterClient] = None):
        """
        Initialize the writing prompt system.
        
        Args:
            openrouter_client: OpenRouter API client
        """
        self.openrouter_client = openrouter_client
        
        # Prompt templates for different sections
        self.prompt_templates = {
            PromptType.INTRODUCTION: {
                "structure": [
                    "Start with a compelling hook that captures the reader's attention",
                    "Provide background context for your research topic",
                    "Clearly state your research problem or question",
                    "Outline the significance and relevance of your study",
                    "Present your thesis statement or main argument",
                    "Preview the structure of your thesis"
                ],
                "content": [
                    "What is the current state of knowledge in your field?",
                    "What gap in the literature does your research address?",
                    "Why is this research important now?",
                    "What are the potential implications of your findings?",
                    "How does your work contribute to the field?"
                ]
            },
            PromptType.LITERATURE_REVIEW: {
                "structure": [
                    "Organize literature thematically rather than chronologically",
                    "Identify key debates and controversies in the field",
                    "Synthesize findings rather than just summarizing",
                    "Highlight gaps and limitations in existing research",
                    "Connect literature to your research questions"
                ],
                "content": [
                    "What are the major theoretical frameworks in your field?",
                    "Which studies are most influential and why?",
                    "What methodological approaches have been used?",
                    "Where do researchers disagree and why?",
                    "What questions remain unanswered?"
                ]
            },
            PromptType.METHODOLOGY: {
                "structure": [
                    "Justify your methodological choices",
                    "Describe your research design clearly",
                    "Explain data collection procedures",
                    "Detail your analysis methods",
                    "Address limitations and ethical considerations"
                ],
                "content": [
                    "Why is this methodology appropriate for your research questions?",
                    "What are the strengths and limitations of your approach?",
                    "How did you ensure validity and reliability?",
                    "What ethical considerations did you address?",
                    "How did you handle potential biases?"
                ]
            },
            PromptType.RESULTS: {
                "structure": [
                    "Present findings objectively without interpretation",
                    "Use clear headings and subheadings",
                    "Include relevant tables, figures, and charts",
                    "Report both significant and non-significant findings",
                    "Organize results by research question or theme"
                ],
                "content": [
                    "What are your key findings?",
                    "Which results were expected and which were surprising?",
                    "How do your findings relate to your research questions?",
                    "What patterns or trends do you observe?",
                    "Are there any unexpected or contradictory results?"
                ]
            },
            PromptType.DISCUSSION: {
                "structure": [
                    "Interpret your findings in context",
                    "Compare results with existing literature",
                    "Discuss implications and significance",
                    "Address limitations honestly",
                    "Suggest directions for future research"
                ],
                "content": [
                    "What do your findings mean in the broader context?",
                    "How do your results support or challenge existing theories?",
                    "What are the practical implications of your findings?",
                    "What are the limitations of your study?",
                    "What questions arise from your research?"
                ]
            },
            PromptType.CONCLUSION: {
                "structure": [
                    "Summarize key findings and contributions",
                    "Restate the significance of your research",
                    "Discuss broader implications",
                    "Acknowledge limitations",
                    "Suggest future research directions"
                ],
                "content": [
                    "What are the main takeaways from your research?",
                    "How has your work advanced the field?",
                    "What are the practical applications?",
                    "What would you do differently?",
                    "What research should be done next?"
                ]
            }
        }
        
        logger.info("Writing prompt system initialized")
    
    async def generate_writing_prompts(self, 
                                     content: str,
                                     prompt_type: PromptType,
                                     category: PromptCategory = PromptCategory.CONTENT,
                                     context: Optional[str] = None) -> WritingPromptResponse:
        """
        Generate contextual writing prompts and suggestions.
        
        Args:
            content: Current content being worked on
            prompt_type: Type of section being written
            category: Category of assistance needed
            context: Additional context information
            
        Returns:
            Writing prompt response with suggestions
        """
        try:
            # Get base prompts from templates
            base_prompts = self._get_base_prompts(prompt_type, category)
            
            # Generate AI-powered suggestions if client available
            if self.openrouter_client:
                ai_suggestions = await self._generate_ai_suggestions(content, prompt_type, category, context)
            else:
                ai_suggestions = []
            
            # Generate research questions
            research_questions = self._generate_research_questions(content, prompt_type)
            
            # Generate citation recommendations
            citation_recommendations = self._generate_citation_recommendations(content, prompt_type)
            
            # Generate style improvements
            style_improvements = self._generate_style_improvements(content)
            
            # Generate next steps
            next_steps = self._generate_next_steps(content, prompt_type)
            
            # Combine all suggestions
            all_suggestions = base_prompts + ai_suggestions
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(content, all_suggestions)
            
            return WritingPromptResponse(
                prompt_type=prompt_type,
                category=category,
                suggestions=all_suggestions[:10],  # Limit to top 10
                research_questions=research_questions[:5],
                citation_recommendations=citation_recommendations[:5],
                style_improvements=style_improvements[:5],
                next_steps=next_steps[:3],
                confidence_score=confidence_score,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate writing prompts: {e}")
            raise Exception(f"Prompt generation failed: {e}")
    
    def _get_base_prompts(self, prompt_type: PromptType, category: PromptCategory) -> List[str]:
        """Get base prompts from templates."""
        section_prompts = self.prompt_templates.get(prompt_type, {})
        category_prompts = section_prompts.get(category.value, [])
        
        # Add general prompts if specific ones not available
        if not category_prompts and category == PromptCategory.CONTENT:
            category_prompts = section_prompts.get("content", [])
        elif not category_prompts and category == PromptCategory.STRUCTURE:
            category_prompts = section_prompts.get("structure", [])
        
        return category_prompts.copy()
    
    async def _generate_ai_suggestions(self, content: str, prompt_type: PromptType, 
                                     category: PromptCategory, context: Optional[str]) -> List[str]:
        """Generate AI-powered writing suggestions."""
        if not self.openrouter_client:
            return []
        
        try:
            # Prepare AI prompt
            ai_prompt = f"""
            As an expert academic writing assistant, analyze the following {prompt_type.value} section content and provide specific, actionable writing suggestions for {category.value} improvement.

            Current content:
            {content[:1000]}...

            Context: {context or 'No additional context provided'}

            Please provide 5 specific, actionable suggestions to improve this {prompt_type.value} section, focusing on {category.value}. Each suggestion should be:
            1. Specific and actionable
            2. Appropriate for academic writing
            3. Relevant to the {prompt_type.value} section
            4. Focused on {category.value} improvement

            Format your response as a numbered list.
            """
            
            response = self.openrouter_client.chat_completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": ai_prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            if response and 'choices' in response and response['choices']:
                ai_response = response['choices'][0]['message']['content']
                
                # Parse numbered list
                suggestions = []
                lines = ai_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # Remove numbering and clean up
                        suggestion = line.split('.', 1)[-1].strip()
                        if suggestion:
                            suggestions.append(suggestion)
                
                return suggestions[:5]
            
        except Exception as e:
            logger.error(f"AI suggestion generation failed: {e}")
        
        return []
    
    def _generate_research_questions(self, content: str, prompt_type: PromptType) -> List[str]:
        """Generate relevant research questions."""
        questions = []
        content_lower = content.lower()
        
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(content)
        
        # Generate questions based on section type
        if prompt_type == PromptType.INTRODUCTION:
            questions.extend([
                f"What is the current understanding of {term}?" for term in key_terms[:2]
            ])
            questions.append("What gap in knowledge does this research address?")
            
        elif prompt_type == PromptType.LITERATURE_REVIEW:
            questions.extend([
                f"How has research on {term} evolved over time?" for term in key_terms[:2]
            ])
            questions.append("What are the main theoretical debates in this field?")
            
        elif prompt_type == PromptType.METHODOLOGY:
            questions.append("What methodological approaches are most appropriate for this research?")
            questions.append("How can validity and reliability be ensured?")
            
        elif prompt_type == PromptType.RESULTS:
            questions.append("What patterns emerge from the data?")
            questions.append("Which findings are most significant and why?")
            
        elif prompt_type == PromptType.DISCUSSION:
            questions.append("How do these findings contribute to existing knowledge?")
            questions.append("What are the broader implications of these results?")
        
        return questions
    
    def _generate_citation_recommendations(self, content: str, prompt_type: PromptType) -> List[str]:
        """Generate citation recommendations."""
        recommendations = []
        
        # Analyze content for citation needs
        citation_indicators = [
            'research shows', 'studies indicate', 'evidence suggests',
            'according to', 'findings reveal', 'data demonstrates'
        ]
        
        content_lower = content.lower()
        found_indicators = [ind for ind in citation_indicators if ind in content_lower]
        
        if found_indicators:
            recommendations.append("Add citations for statements that reference research findings")
            recommendations.append("Ensure all factual claims are properly supported")
        
        # Section-specific recommendations
        if prompt_type == PromptType.LITERATURE_REVIEW:
            recommendations.append("Include seminal works in your field")
            recommendations.append("Cite recent studies to show current relevance")
            
        elif prompt_type == PromptType.METHODOLOGY:
            recommendations.append("Cite methodological sources to justify your approach")
            recommendations.append("Reference validation studies for your instruments")
            
        elif prompt_type == PromptType.DISCUSSION:
            recommendations.append("Compare your findings with previous research")
            recommendations.append("Cite studies that support or contradict your results")
        
        return recommendations
    
    def _generate_style_improvements(self, content: str) -> List[str]:
        """Generate style improvement suggestions."""
        improvements = []
        
        # Analyze writing style
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 25:
            improvements.append("Consider breaking down long sentences for better readability")
        
        # Check for passive voice (simplified)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(content.lower().count(ind) for ind in passive_indicators)
        
        if passive_count > len(content.split()) * 0.1:  # More than 10% passive indicators
            improvements.append("Consider using more active voice constructions")
        
        # Check for academic tone
        informal_words = ['really', 'very', 'quite', 'pretty', 'kind of', 'sort of']
        if any(word in content.lower() for word in informal_words):
            improvements.append("Replace informal language with more academic alternatives")
        
        # Check for transitions
        if len(content.split('\n\n')) > 2:  # Multiple paragraphs
            transition_words = ['however', 'furthermore', 'moreover', 'therefore', 'consequently']
            if not any(word in content.lower() for word in transition_words):
                improvements.append("Add transitional phrases to improve flow between ideas")
        
        improvements.append("Ensure consistent terminology throughout the section")
        
        return improvements
    
    def _generate_next_steps(self, content: str, prompt_type: PromptType) -> List[str]:
        """Generate next steps suggestions."""
        steps = []
        
        # Analyze content completeness
        word_count = len(content.split())
        
        if word_count < 200:
            steps.append("Expand the content with more detailed explanations")
        
        # Section-specific next steps
        if prompt_type == PromptType.INTRODUCTION:
            steps.append("Develop a clear thesis statement")
            steps.append("Add background context for your research")
            
        elif prompt_type == PromptType.LITERATURE_REVIEW:
            steps.append("Synthesize findings from multiple sources")
            steps.append("Identify gaps in the current literature")
            
        elif prompt_type == PromptType.METHODOLOGY:
            steps.append("Justify your methodological choices")
            steps.append("Address potential limitations")
            
        elif prompt_type == PromptType.RESULTS:
            steps.append("Add visual representations of key findings")
            steps.append("Organize results by research question")
            
        elif prompt_type == PromptType.DISCUSSION:
            steps.append("Connect findings to broader theoretical frameworks")
            steps.append("Discuss practical implications")
        
        return steps
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content."""
        import re
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[A-Za-z]{4,}\b', content)
        
        # Filter common words
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'them', 'their', 'there',
            'where', 'when', 'what', 'which', 'while', 'would', 'could',
            'should', 'might', 'will', 'shall', 'must', 'have', 'been',
            'were', 'was', 'are', 'is', 'am', 'be', 'do', 'does', 'did'
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words and len(word) > 4:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Return most frequent terms
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [term[0] for term in sorted_terms[:10]]
    
    def _calculate_confidence_score(self, content: str, suggestions: List[str]) -> float:
        """Calculate confidence score for suggestions."""
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on content length
        word_count = len(content.split())
        if word_count > 100:
            confidence += 0.2
        if word_count > 500:
            confidence += 0.1
        
        # Increase confidence based on number of suggestions
        if len(suggestions) >= 5:
            confidence += 0.1
        
        # Increase confidence if AI suggestions are available
        if self.openrouter_client:
            confidence += 0.1
        
        return min(1.0, confidence)
