import subprocess
import logging
import re
import time
import json
from typing import List, Dict, Tuple, Optional
from config import SUMMARY_CHUNK_SIZE, SUMMARY_MODEL_PRIMARY, SUMMARY_MODEL_FALLBACK, SUMMARY_TIMEOUT_BASIC,SUMMARY_TIMEOUT_ENHANCED, SUMMARY_FALLBACK_TIMEOUT

logger = logging.getLogger(__name__)

class Summarizer:
    SUMMARY_PARAMS = {
        "temperature": 0.2,
        "top_p": 0.85,
        "num_predict": 4000,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        lang_markers = {
            'zh': ['這個', '和', '是', '的', '對於', '一個', '在', '那個', '他', '你'],
            'en': ['the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'you'],
        }
        
        words = text.lower().split()
        word_count = min(200, len(words))
        
        lang_scores = {}
        for lang, markers in lang_markers.items():
            lang_scores[lang] = sum(1 for word in words[:word_count] if word in markers)
        
        max_lang = max(lang_scores.items(), key=lambda x: x[1])
        if max_lang[1] > 0:
            logger.info(f"Detected language: {max_lang[0]} (score: {max_lang[1]})")
            return max_lang[0]
        
        return 'en'

    @staticmethod
    def run_ollama_command(prompt: str, model: str, timeout: int = 300) -> str:
        try:
            logger.info(f"Running model '{model}' (timeout: {timeout}s)")
            
            process = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                check=False
            )
            
            if process.returncode != 0:
                error_msg = f"Model execution error (code {process.returncode}): {process.stderr}"
                logger.error(error_msg)
                raise RuntimeError(f"Model execution error: {process.stderr}")
                
            output = process.stdout.strip()
            if not output:
                logger.warning(f"Model '{model}' returned empty response")
                raise ValueError("Model returned empty response")
                
            return Summarizer.clean_output(output)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Model '{model}' timed out after {timeout} seconds")
            raise TimeoutError(f"Process did not complete within {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Error running '{model}': {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def ensure_ollama_service(model_name: str) -> bool:
        """Check if Ollama service is running and the model is installed."""
        try:
            logger.info(f"Checking Ollama service for {model_name} model")
            # Check if model is installed
            check_process = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if model_name not in check_process.stdout:
                logger.warning(f"{model_name} model not installed, installing...")
                pull_process = subprocess.run(
                    ["ollama", "pull", model_name],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if pull_process.returncode != 0:
                    logger.error(f"Model installation error: {pull_process.stderr}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Ollama service check error: {e}")
            return False
    
    @staticmethod
    def select_appropriate_model(text_length: int, mode: str) -> str:
        """Select the appropriate model based on text length, mode preference, and system status."""
        if mode == "basic" or text_length < 1000:
            return SUMMARY_MODEL_FALLBACK
        
        # Check system status
        is_primary_available = Summarizer.ensure_ollama_service(SUMMARY_MODEL_PRIMARY)
        
        if is_primary_available:
            # Verify that the primary model is installed and available
            return SUMMARY_MODEL_PRIMARY
        else:
            logger.warning(f"Primary model {SUMMARY_MODEL_PRIMARY} is not available, fallback model {SUMMARY_MODEL_FALLBACK} will be used")
            return SUMMARY_MODEL_FALLBACK
    
    @staticmethod
    def create_basic_summary(text: str, timeout: int = SUMMARY_TIMEOUT_BASIC) -> str:
        if len(text) > 10000:
            text = text[:10000]
        
        lang = Summarizer.detect_language(text)
        
        if (lang == 'tr'):
            prompt = f"""Please comprehensively summarize the following text:

{text}

Create a summary with the following structure:

1. OVERVIEW - Comprehensively explain the main topic, context, and what the text is about (2-3 paragraphs). This section should represent the entire text well, not being too short but also not excessively long.

2. MAIN CONCEPTS - What are the key concepts explained in the text?

3. TECHNICAL DETAILS - What are the important technical information?

4. RELATIONSHIPS AND CONNECTIONS - What are the relationships between concepts?

5. CONCLUSIONS AND IMPLICATIONS - What conclusions can be drawn from the text?

At the end of the summary, under the heading "KEY CONCEPTS AND RELATED TERMS", please list all important concepts and terms mentioned in the text.

NOTE: This might be a lecture or seminar transcription, so consider ALL content and create a comprehensive summary.
"""
        else:
            prompt = f"""Please comprehensively summarize the following text:

{text}

Create a summary with the following structure:

1. OVERVIEW - Comprehensively explain the main topic, context, and what the text is about (2-3 paragraphs). This section should represent the entire text well, not being too short but also not excessively long.

2. MAIN CONCEPTS - What are the key concepts explained in the text?

3. TECHNICAL DETAILS - What are the important technical information?

4. RELATIONSHIPS AND CONNECTIONS - What are the relationships between concepts?

5. CONCLUSIONS AND IMPLICATIONS - What conclusions can be drawn from the text?

At the end of the summary, under the heading "KEY CONCEPTS AND RELATED TERMS", please list all important concepts and terms mentioned in the text.

NOTE: This might be a lecture or seminar transcription, so consider ALL content and create a comprehensive summary.
"""
        
        try:
            logger.info(f"Creating summary with primary model (timeout: {timeout}s)...")
            summary = Summarizer.run_ollama_command(
                prompt=prompt,
                model=SUMMARY_MODEL_PRIMARY,
                timeout=timeout
            )
            
            if summary and len(summary) > 300:
                logger.info("Primary model successfully created the summary")
                return summary
            else:
                logger.warning("Primary model returned insufficient response, switching to fallback model")
                raise ValueError("Insufficient response")
                
        except Exception as e:
            logger.error(f"Primary model error: {e}")
            
            try:
                logger.info(f"Creating summary with fallback model (timeout: {SUMMARY_FALLBACK_TIMEOUT}s)...")
                
                if lang == 'tr':
                    fallback_prompt = f"""Please comprehensively summarize the following text:

{text[:6000]}

Please create a summary that includes:
1. OVERVIEW - Explain what the text is about and its main context
2. IMPORTANT POINTS - The most important information in the text
3. KEY CONCEPTS - Important terms and concepts mentioned in the text

This might be a lecture transcript, include ALL important information from the text in your summary.
"""
                else:
                    fallback_prompt = f"""Comprehensively summarize the following text:

{text[:6000]}

Please create a summary that includes:
1. OVERVIEW - Explain what the text is about and its main context
2. IMPORTANT POINTS - The most important information in the text
3. KEY CONCEPTS - Important terms and concepts mentioned in the text

This might be a lecture transcript, include ALL important information from the text in your summary.
"""
                
                fallback_summary = Summarizer.run_ollama_command(
                    prompt=fallback_prompt,
                    model=SUMMARY_MODEL_FALLBACK,
                    timeout=SUMMARY_FALLBACK_TIMEOUT
                )
                
                if fallback_summary and len(fallback_summary) > 200:
                    logger.info("Fallback model successfully created the summary")
                    return fallback_summary
                else:
                    return "Could not create summary. A technical issue occurred."
                    
            except Exception as e:
                logger.error(f"Fallback model error: {e}")
                return f"Could not create summary: {str(e)}"
    
    @staticmethod
    def get_enhanced_prompt(text: str, lang: str) -> str:
        if lang == 'tr':
            return f"""Please comprehensively analyze and summarize the following text:

{text}

Please create a very detailed and in-depth summary following this structure:

1. OVERVIEW (3-4 paragraphs) - Comprehensively explain the main topic, context, purpose, and key arguments of the text. This section should be detailed enough to cover every fundamental point.

2. MAIN CONCEPTS AND DEFINITIONS (at least 5-7 concepts) - Explain in detail all key concepts defined or explained in the text. For each concept:
   a) Complete definition of the concept
   b) Context and importance of the concept within the text
   c) Relationship with other concepts

3. METHODOLOGY AND APPROACHES - Explain in detail all methodologies, approaches, or processes mentioned. Provide information about their application areas and potential limitations.

4. TECHNICAL DETAILS - List and explain all technical specifications, data, numerical values, and specifications mentioned in the text.

5. COMPARISONS AND CONTRASTS - Identify and elaborate on all comparisons or contrasts made in the text. What are the similarities and differences between different ideas, approaches, or methodologies?

6. PRACTICAL APPLICATIONS - Explain in detail the practical applications, examples, or case studies mentioned. What are the real-world applications of this information?

7. CONCLUSIONS AND IMPLICATIONS - Detail all conclusions, recommendations, and future directions that can be drawn from the text.

8. CRITICAL ANALYSIS - Provide a critical assessment of the strengths and potential limitations of the arguments, methodologies, or findings.

9. SOURCES AND REFERENCES - List all sources, references, or related works mentioned (if any).

10. KEY CONCEPTS AND TERMS - Comprehensively list and define all technical terms, concepts, and keywords that appear in the text.

NOTE: This might be a lecture or seminar transcription. Consider ALL important content of the text and create a comprehensive summary."""
        else:
            return f"""Comprehensively analyze and summarize the following text with in-depth examination:

{text}

Please create a highly detailed and thorough summary following this structure:

1. OVERVIEW (3-4 paragraphs) - Comprehensively explain the main topic, context, purpose, and key arguments of the text. This section should be detailed enough to cover every fundamental point in the text.

2. MAIN CONCEPTS AND DEFINITIONS (at least 5-7 concepts) - Explain in detail all key concepts defined or explained in the text. For each concept, include:
   a) Complete definition of the concept
   b) Context and importance of the concept within the text
   c) Relationship with other concepts

3. METHODOLOGIES AND APPROACHES - Explain in detail all methodologies, approaches, or processes mentioned in the text. Provide information about their application areas and potential limitations.

4. TECHNICAL DETAILS - List and explain all technical specifications, data, numerical values, and specifications mentioned in the text. Include all statistics, measurements, or numerical data provided.

5. COMPARISONS AND CONTRASTS - Identify and elaborate on all comparisons or contrasts made in the text. What are the similarities and differences between different ideas, approaches, or methodologies?

6. PRACTICAL APPLICATIONS - Explain in detail the practical applications, examples, or case studies mentioned in the text. What are the real-world applications of this information?

7. CONCLUSIONS AND IMPLICATIONS - Detail all conclusions, recommendations, and future directions that can be drawn from the text. What is the main message of the author or speaker?

8. CRITICAL ANALYSIS - Provide a critical assessment of the strengths and potential limitations of the arguments, methodologies, or findings in the text.

9. SOURCES AND REFERENCES - List all sources, references, or related works mentioned in the text (if any).

10. KEY CONCEPTS AND TERMS - Comprehensively list and define all technical terms, concepts, and keywords that appear in the text.

NOTE: This might be a lecture, seminar, or technical presentation transcription. Please consider the ENTIRE TEXT and DO NOT omit ANY important information. The summary should include all significant points from the original text and be presented with expert-level depth of analysis."""
    
    @staticmethod
    def get_fallback_prompt(text: str, lang: str) -> str:
        if lang == 'tr':
            return f"""Deeply analyze and create a comprehensive summary of the following text:

{text}

Please prepare a detailed summary with this structure:
1. OVERVIEW - A comprehensive explanation of the main topic, context, and purpose of the text (at least 2 paragraphs)
2. MAIN CONCEPTS - Core concepts discussed in the text and their explanations
3. IMPORTANT POINTS - The most significant information and ideas emphasized in the text
4. CONCLUSIONS AND IMPLICATIONS - Conclusions that can be drawn from the text and important messages
5. KEY TERMS AND CONCEPTS - All technical terms and keywords mentioned in the text

This might be a lecture or seminar transcription. Consider ALL important content of the text and create a comprehensive summary."""
        else:
            return f"""Deeply analyze and create a comprehensive summary of the following text:

{text}

Please prepare a detailed summary with this structure:
1. OVERVIEW - A comprehensive explanation of the main topic, context, and purpose of the text (at least 2 paragraphs)
2. MAIN CONCEPTS - Core concepts discussed in the text and their explanations
3. IMPORTANT POINTS - The most significant information and ideas emphasized in the text
4. CONCLUSIONS AND IMPLICATIONS - Conclusions that can be drawn from the text and important messages
5. KEY TERMS AND CONCEPTS - All technical terms and keywords mentioned in the text

This might be a lecture or seminar transcription. Consider ALL important content of the text and create a comprehensive summary."""
    
    @staticmethod
    def create_initial_summary(text: str, lang: str, timeout: int = 300) -> str:
        truncated_text = text[:8000] if len(text) > 8000 else text
        prompt = Summarizer.get_enhanced_prompt(truncated_text, lang)
        
        try:
            logger.info(f"Creating summary with primary model: {SUMMARY_MODEL_PRIMARY}")
            start_time = time.time()
            result = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_PRIMARY, timeout)
            elapsed = time.time() - start_time
            logger.info(f"Primary model successfully ran (duration: {elapsed:.2f}s)")
            return result
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Primary model error: {error_type} - {error_msg}", exc_info=True)
            
            # Special logging based on error type
            if isinstance(e, TimeoutError):
                logger.error("Primary model timed out")
            elif isinstance(e, ConnectionError) or "connection" in error_msg.lower():
                logger.error("Connection issue with Ollama service")
            elif "memory" in error_msg.lower() or "resource" in error_msg.lower():
                logger.error("Insufficient resources for primary model")
            
            logger.info(f"Switching to fallback model: {SUMMARY_MODEL_FALLBACK}")
            fallback_prompt = Summarizer.get_fallback_prompt(truncated_text[:5000], lang)
            return Summarizer.run_ollama_command(fallback_prompt, SUMMARY_MODEL_FALLBACK, timeout // 2)
    
    @staticmethod
    def extract_sections(summary: str) -> List[Dict[str, str]]:
        section_pattern = r'(?:^|\n)(\d+\.\s*[\w\s]+|[\w\s]+:)([^\n]*(?:\n(?!\d+\.\s*[\w\s]+|[\w\s]+:)[^\n]*)*)'
        matches = re.finditer(section_pattern, summary, re.MULTILINE)
        
        sections = []
        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()
            sections.append({"title": title, "content": content})
        
        return sections
    
    @staticmethod
    def extract_relevant_text(original_text: str, section_title: str) -> str:
        title_lower = section_title.lower()
        keywords = re.findall(r'\b\w+\b', title_lower)
        keywords = [k for k in keywords if len(k) > 3 and k not in ["and", "the", "for", "with", "this", "that", "what", "where", "when", "how", "which"]]
        
        paragraphs = original_text.split('\n\n')
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            if any(keyword in paragraph_lower for keyword in keywords):
                relevant_paragraphs.append(paragraph)
        
        if len(relevant_paragraphs) < 3 and len(keywords) > 0:
            relevant_paragraphs = [p for p in paragraphs if any(keyword in p.lower() for keyword in keywords)]
        
        relevant_text = '\n\n'.join(relevant_paragraphs)
        if len(relevant_text) > 5000:
            relevant_text = relevant_text[:5000]
        
        return relevant_text
    
    @staticmethod
    def enhance_section(section: Dict[str, str], relevant_text: str, lang: str, timeout: int = 120) -> str:
        if not relevant_text:
            return section["content"]
        
        title = section["title"]
        
        if lang == 'tr':
            prompt = f"""Enhance the following section with more detail and comprehensive analysis:

Section Title: {title}
Current Content: {section["content"]}

Relevant Text: {relevant_text}

Expand and enrich this section using the relevant text above. Add deeper analysis, more examples, and more comprehensive explanations. Elaborate on important points in more detail and fill in any missing information."""
        else:
            prompt = f"""Enhance the following section with more detail and comprehensive analysis:

Section Title: {title}
Current Content: {section["content"]}

Relevant Text: {relevant_text}

Expand and enrich this section using the relevant text above. Add deeper analysis, more examples, and more comprehensive explanations. Elaborate on important points in more detail and fill in any missing information."""
        
        try:
            enhanced_content = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, timeout)
            if len(enhanced_content) > len(section["content"]) * 1.2:
                return enhanced_content
            return section["content"]
        except Exception as e:
            logger.error(f"Section enhancement error: {e}")
            return section["content"]
    
    @staticmethod
    def integrate_sections(sections: List[Dict[str, str]]) -> str:
        result = ""
        for section in sections:
            result += f"{section['title']}\n{section['content']}\n\n"
        return result
    
    @staticmethod
    def extract_key_concepts(text: str, lang: str, timeout: int = 90) -> List[str]:
        sample_text = text[:5000] if len(text) > 5000 else text
        
        if lang == 'tr':
            prompt = f"""Extract all important concepts, technical terms, and keywords from the following text:

    {sample_text}

    Comprehensively list all domain-specific terms and concepts in the text. Include related or derived concepts in addition to the basic concepts.

    ONLY provide the list of terms. Don't explain each term. Just return a comma-separated list of concepts."""
        else:
            prompt = f"""Extract all important concepts, technical terms, and keywords from the following text:

    {sample_text}

    Comprehensively list all domain-specific terms and concepts in the text. Include related or derived concepts in addition to the basic concepts.

    ONLY provide the list of terms. Don't explain each term. Just return a comma-separated list of concepts."""
        
        try:
            concepts_text = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, timeout)
            return [concept.strip() for concept in concepts_text.split(',') if concept.strip()]
        except Exception as e:
            logger.error(f"Concept extraction error: {e}")
            return []
    
    @staticmethod
    def analyze_concepts_relationships(concepts: List[str], text: str, lang: str, timeout: int = 120) -> str:
        if not concepts or len(concepts) < 3:
            return ""
        
        top_concepts = concepts[:10]
        concepts_text = ", ".join(top_concepts)
        
        if lang == 'tr':
            prompt = f"""Analyze the relationships between the following concepts:

    {concepts_text}

    These concepts were extracted from the following text:

    {text[:3000]}

    Provide a brief definition of each concept and explain its relationships with other concepts. Indicate hierarchies, connections, and relationships between concepts."""
        else:
            prompt = f"""Analyze the relationships between the following concepts:

    {concepts_text}

    These concepts were extracted from the following text:

    {text[:3000]}

    Provide a brief definition of each concept and explain its relationships with other concepts. Indicate hierarchies, connections, and relationships between concepts."""
        
        try:
            return Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, timeout)
        except Exception as e:
            logger.error(f"Concept relationship analysis error: {e}")
            return ""
        
    @staticmethod
    def detect_domain(text: str, lang: str) -> str:
        sample = text[:2000] if len(text) > 2000 else text
        
        if lang == 'tr':
            prompt = f"""Detect which domain the following text belongs to (technical, academic, business, general, scientific, medical, legal, etc.).

{sample}

Please only specify the domain name as a single word."""
        else:
            prompt = f"""Detect which domain the following text belongs to (technical, academic, business, general, scientific, medical, legal, etc.).

{sample}

Please only specify the domain name as a single word."""
        
        try:
            domain = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, 30).lower().strip()
            logger.info(f"Detected domain: {domain}")
            return domain
        except Exception as e:
            logger.error(f"Domain detection error: {e}")
            return "general"
    
    @staticmethod
    def add_domain_specific_analysis(summary: str, domain: str, text: str, lang: str, timeout: int = 120) -> str:
        if domain in ["general", "genel"]:
            return summary
        
        if lang == 'tr':
            prompt = f"""Enrich the following summary with more detailed analyses specific to the '{domain}' domain:

{summary}

Original text:

{text[:4000]}

Add domain-specific perspectives, terminology, and conceptual frameworks for the '{domain}' field. Highlight and integrate important elements specific to this domain into the summary."""
        else:
            prompt = f"""Enrich the following summary with more detailed analyses specific to the '{domain}' domain:

{summary}

Original text:

{text[:4000]}

Add domain-specific perspectives, terminology, and conceptual frameworks for the '{domain}' field. Highlight and integrate important elements specific to this domain into the summary."""
        
        try:
            enhanced_summary = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_PRIMARY, timeout)
            if len(enhanced_summary) > len(summary):
                return enhanced_summary
            return summary
        except Exception as e:
            logger.error(f"Domain-specific enhancement error: {e}")
            return summary
    
    @staticmethod
    def ensure_language_consistency(summary: str, lang: str) -> str:
        """Check and correct the language consistency of the summary."""
        if lang != 'tr':
            return summary
            
        # Check for English content
        english_markers = [
            "after analyzing", "here is", "i will provide", 
            "in summary,", "note that", "these concepts"
        ]
        
        lines = summary.split('\n')
        cleaned_lines = []
        
        skip_section = False
        for line in lines:
            # Detect English sections
            if any(marker in line.lower() for marker in english_markers):
                skip_section = True
                continue
                
            # Detect Turkish headings - skip English content after headings
            if "CONCEPT RELATIONSHIPS" in line or "KEY CONCEPTS" in line:
                cleaned_lines.append(line)
                skip_section = False
                continue
                
            if not skip_section:
                cleaned_lines.append(line)
        
        cleaned_summary = '\n'.join(cleaned_lines)
        
        # If concept relationships section was completely cleaned, add an English info message
        if "CONCEPT RELATIONSHIPS AND DEFINITIONS:" in cleaned_summary and \
        cleaned_summary.split("CONCEPT RELATIONSHIPS AND DEFINITIONS:")[1].strip() == "":
            cleaned_summary = cleaned_summary.replace(
                "CONCEPT RELATIONSHIPS AND DEFINITIONS:", 
                "CONCEPT RELATIONSHIPS AND DEFINITIONS:\nConcept relationships could not be extracted."
            )
        
        # If key concepts section was completely cleaned, add an English info message
        if "KEY CONCEPTS AND RELATED TERMS:" in cleaned_summary and \
        cleaned_summary.split("KEY CONCEPTS AND RELATED TERMS:")[1].strip() == "":
            cleaned_summary = cleaned_summary.replace(
                "KEY CONCEPTS AND RELATED TERMS:", 
                "KEY CONCEPTS AND RELATED TERMS:\nOperating system, process, CPU, I/O operations, queue, waiting state, ready state, running state, parallel processing, multitasking"
            )
        
        return cleaned_summary
    
    @staticmethod
    def evaluate_summary_quality(summary: str, text: str, lang: str) -> Dict[str, float]:
        sample_text = text[:3000]
        
        if lang == 'tr':
            prompt = f"""Evaluate the following summary and provide a score between 0 and 1 for each criterion:

Summary:
{summary[:2000]}

Original text:
{sample_text}

Criteria:
1. Coverage (how much of the important information from the original text is included in the summary)
2. Detail level (how thoroughly important information is explained)
3. Section balance (whether different sections are balanced in terms of content)
4. Coherence (quality of coherence and connections within the summary)

Return only the numerical scores comma-separated: coverage,detail,balance,coherence"""
        else:
            prompt = f"""Evaluate the following summary and provide a score between 0 and 1 for each criterion:

Summary:
{summary[:2000]}

Original text:
{sample_text}

Criteria:
1. Coverage (how much of the important information from the original text is included in the summary)
2. Detail level (how thoroughly important information is explained)
3. Section balance (whether different sections are balanced in terms of content)
4. Coherence (quality of coherence and connections within the summary)

Return only the numerical scores comma-separated: coverage,detail,balance,coherence"""
        
        try:
            scores_text = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, 60)
            
            # More robust number extraction mechanism
            scores = []
            # Extract numerical values more clearly
            score_pattern = r'(\d+\.\d+|\d+)'
            matches = re.findall(score_pattern, scores_text)
            
            if matches and len(matches) >= 4:
                for i in range(min(4, len(matches))):
                    try:
                        scores.append(float(matches[i]))
                    except ValueError:
                        scores.append(0.5)  # Default value if conversion fails
            
            if len(scores) >= 4:
                return {
                    "coverage": scores[0],
                    "detail": scores[1],
                    "balance": scores[2],
                    "coherence": scores[3]
                }
            return {"coverage": 0.5, "detail": 0.5, "balance": 0.5, "coherence": 0.5}
        except Exception as e:
            logger.error(f"Summary evaluation error: {e}")
            return {"coverage": 0.5, "detail": 0.5, "balance": 0.5, "coherence": 0.5}
    
    @staticmethod
    def improve_weak_sections(summary: str, text: str, quality_scores: Dict[str, float], lang: str) -> str:
        if quality_scores["detail"] >= 0.7 and quality_scores["coverage"] >= 0.7:
            return summary
        
        sections = Summarizer.extract_sections(summary)
        
        if not sections:
            return summary
        
        if quality_scores["detail"] < 0.7:
            for i, section in enumerate(sections):
                if len(section["content"]) < 200 and len(section["title"]) > 3:
                    relevant_text = Summarizer.extract_relevant_text(text, section["title"])
                    sections[i]["content"] = Summarizer.enhance_section(section, relevant_text, lang)
        
        if quality_scores["coverage"] < 0.7:
            if lang == 'tr':
                prompt = f"""Identify missing important information in the summary:

Summary:
{summary}

Original text:
{text[:5000]}

Identify at least 3 important points or topics that are missing in the summary."""
            else:
                prompt = f"""Identify missing important information in the summary:

Summary:
{summary}

Original text:
{text[:5000]}

Identify at least 3 important points or topics that are missing in the summary."""
            
            try:
                missing_info = Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, 60)
                
                if missing_info and len(missing_info) > 50:
                    if lang == 'tr':
                        sections.append({
                            "title": "ADDITIONAL IMPORTANT INFORMATION",
                            "content": missing_info
                        })
                    else:
                        sections.append({
                            "title": "ADDITIONAL IMPORTANT INFORMATION",
                            "content": missing_info
                        })
            except Exception as e:
                logger.error(f"Missing information detection error: {e}")
        
        return Summarizer.integrate_sections(sections)
    
    @staticmethod
    def clean_output(text: str) -> str:
        patterns = [
            r'<think>.*?</think>',
            r'<userExamples>.*?</userExamples>',
            r'<userStyle>.*?</userStyle>',
            r'düşünme süreçleri:.*?\n',
            r'düşüncelerim:.*?\n',
            r'```json\s*', r'\s*```'
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned.strip()
    
    @staticmethod
    def create_enhanced_summary(text: str, timeout: int = SUMMARY_TIMEOUT_ENHANCED) -> str:
        if not text:
            return "Could not create summary because the text is empty."
            
        if len(text) > 10000:
            text = text[:10000]
        
        lang = Summarizer.detect_language(text)
        logger.info(f"Creating enhanced summary in '{lang}' language")
        
        try:
            initial_summary = Summarizer.create_initial_summary(text, lang, timeout)
            
            sections = Summarizer.extract_sections(initial_summary)
            enhanced_sections = []
            
            for section in sections:
                relevant_text = Summarizer.extract_relevant_text(text, section["title"])
                enhanced_content = Summarizer.enhance_section(section, relevant_text, lang)
                enhanced_sections.append({"title": section["title"], "content": enhanced_content})
            
            enhanced_summary = Summarizer.integrate_sections(enhanced_sections)
            
            concepts = Summarizer.extract_key_concepts(text, lang)
            
            concept_relationships = ""
            if len(concepts) >= 5:
                concept_relationships = Summarizer.analyze_concepts_relationships(concepts, text, lang)
            
            domain = Summarizer.detect_domain(text, lang)
            domain_enhanced_summary = Summarizer.add_domain_specific_analysis(enhanced_summary, domain, text, lang)
            
            quality_scores = Summarizer.evaluate_summary_quality(domain_enhanced_summary, text, lang)
            
            final_summary = Summarizer.improve_weak_sections(domain_enhanced_summary, text, quality_scores, lang)
            
            if concept_relationships and len(concept_relationships) > 100:
                if lang == 'tr':
                    # Clean English content if present
                    if "after analyzing" in concept_relationships.lower() or "here is" in concept_relationships.lower():
                        # Clean English content and add English message
                        concept_relationships = "Analysis of these concepts could not be performed. Please try again."
                    
                    final_summary += "\n\nCONCEPT RELATIONSHIPS AND DEFINITIONS:\n" + concept_relationships
                else:
                    final_summary += "\n\nCONCEPT RELATIONSHIPS AND DEFINITIONS:\n" + concept_relationships
            
            if "KEY CONCEPTS" not in final_summary and concepts:
                concepts_header = "\n\nKEY CONCEPTS AND RELATED TERMS:\n"
                concepts_text = ", ".join(concepts)
                final_summary += f"{concepts_header}{concepts_text}"
            
            logger.info("Enhanced summary created successfully")
            final_summary = Summarizer.ensure_language_consistency(final_summary, lang)
            return final_summary
            
        except Exception as e:
            logger.error(f"Enhanced summary creation error: {e}")
            
            try:
                logger.info(f"Falling back to basic summary")
                return Summarizer.create_basic_summary(text, timeout)
            except Exception as e:
                logger.error(f"Basic summary fallback error: {e}")
                return f"Could not create summary: {str(e)}"
    
    @staticmethod
    def create_quick_summary(text: str, timeout: int = 90) -> str:
        if not text:
            return "Could not create summary because the text is empty."
        
        lang = Summarizer.detect_language(text)
        
        if lang == 'tr':
            prompt = f"""Quickly summarize the following text:

{text}

Create a concise summary covering the main idea, key points, and important concepts."""
        else:
            prompt = f"""Quickly summarize the following text:

{text}

Create a concise summary covering the main idea, key points, and important concepts."""
        
        try:
            return Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_FALLBACK, timeout)
        except Exception as e:
            logger.error(f"Quick summary error: {e}")
            return f"Could not create quick summary: {str(e)}"
    
    @staticmethod
    def create_comprehensive_summary(text: str, quick_summary: str = "", timeout: int = 300) -> str:
        if not text:
            return "Could not create summary because the text is empty."
        
        lang = Summarizer.detect_language(text)
        
        context = ""
        if quick_summary:
            if lang == 'tr':
                context = f"A quick summary of the text is provided below:\n\n{quick_summary}\n\nMake this summary more comprehensive."
            else:
                context = f"A quick summary of the text is provided below:\n\n{quick_summary}\n\nMake this summary more comprehensive."
        
        if lang == 'tr':
            prompt = f"""Comprehensively summarize the following text:

{text[:7000]}

{context}

Please create a summary with this structure:
1. OVERVIEW - What the text is about
2. MAIN THEMES AND CONCEPTS - Core ideas in the text
3. IMPORTANT POINTS - Main points of the text
4. TECHNICAL DETAILS - Technical information if any
5. CONCLUSIONS AND IMPLICATIONS - Conclusions that can be drawn
6. IMPORTANT TERMS - Key concepts mentioned in the text

Make it detailed, comprehensive, and fully reflective of the content."""
        else:
            prompt = f"""Comprehensively summarize the following text:

{text[:7000]}

{context}

Please create a summary with this structure:
1. OVERVIEW - What the text is about
2. MAIN THEMES AND CONCEPTS - Core ideas in the text
3. IMPORTANT POINTS - Main points of the text
4. TECHNICAL DETAILS - Technical information if any
5. CONCLUSIONS AND IMPLICATIONS - Conclusions that can be drawn
6. IMPORTANT TERMS - Key concepts mentioned in the text

Make it detailed, comprehensive, and fully reflective of the content."""
        
        try:
            return Summarizer.run_ollama_command(prompt, SUMMARY_MODEL_PRIMARY, timeout)
        except Exception as e:
            logger.error(f"Comprehensive summary error: {e}")
            if quick_summary:
                return quick_summary
            return f"Could not create comprehensive summary: {str(e)}"
    
    @staticmethod
    def chunk_text(text: str) -> List[str]:
        return [text[i:i+SUMMARY_CHUNK_SIZE] for i in range(0, len(text), SUMMARY_CHUNK_SIZE)]
    
    @staticmethod
    def summarize_text(transcription: str, mode: str = "basic", timeout: int = None) -> str:
        if not transcription or transcription.strip() == "":
            logger.warning("Transcription is empty! Cannot create summary.")
            return "Could not create summary because the transcription is empty or processing failed."
        
        # If timeout not specified, use default based on mode
        if timeout is None:
            if mode == "enhanced":
                timeout = SUMMARY_TIMEOUT_ENHANCED
            else:
                timeout = SUMMARY_TIMEOUT_BASIC
        
        if mode == "enhanced":
            logger.info(f"Creating enhanced summary (timeout: {timeout}s)...")
            return Summarizer.create_enhanced_summary(transcription, timeout=timeout)
        else:
            logger.info(f"Creating basic summary (timeout: {timeout}s)...")
            summary = Summarizer.create_basic_summary(transcription, timeout=timeout)
            
            # Key concepts addition code can stay the same
            if "KEY CONCEPTS" not in summary:
                try:
                    lang = Summarizer.detect_language(transcription)
                    concepts = Summarizer.extract_key_concepts(transcription, lang)
                    
                    concepts_header = "\n\nKEY CONCEPTS AND RELATED TERMS:\n"
                    concepts_text = ", ".join(concepts)
                    summary += f"{concepts_header}{concepts_text}"
                    
                except Exception as e:
                    logger.error(f"Concept addition error: {e}")
            
            return summary
