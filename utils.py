from prompts import PODCAST_CONVERSION_PROMPT, SUMMARIZATION_PROMPT
import re
import requests
from io import BytesIO
from PyPDF2 import PdfReader

def create_summarization_messages(content):
    return [
        {
            "role": "system",
            "content": SUMMARIZATION_PROMPT
        },
        {
            "role": "user",
            "content": f"""Please analyze and summarize the following content for podcast creation:

                            ---CONTENT START---
                            {content}
                            ---CONTENT END---

                            Create a comprehensive analysis that identifies:
                            - The most important and interesting aspects
                            - Key insights and surprising facts that would engage podcast listeners  
                            - Complex concepts that need clear explanation
                            - Relatable examples and potential analogies
                            - Thought-provoking questions this content raises or answers
                            - Why this matters to a general audience

                            Structure your analysis clearly so it can be easily used to create an engaging podcast discussion."""
        }
    ]

def create_podcast_conversion_messages(summary, host1="Alex", host2="Romen"):
    return [
        {
            "role": "system", 
            "content": PODCAST_CONVERSION_PROMPT.replace("Alex", host1).replace("Romen", host2)
        },
        {
            "role": "user",
            "content": f"""Transform this content summary into an engaging podcast script:

                            ---SUMMARY START---
                            {summary}
                            ---SUMMARY END---

                            Please follow these steps:

                            1. **<scratchpad>** - Plan how to transform the summary insights into natural conversation, including the best analogies, examples, and discussion flow

                            2. **Generate the full podcast dialogue** between {host1} and {host2} that:
                            - Opens with a compelling hook based on the summary's most engaging aspects
                            - Incorporates the key insights and fascinating facts naturally into conversation
                            - Uses suggested analogies and examples to explain complex concepts
                            - Maintains authentic, natural conversation throughout
                            - Builds complexity gradually while staying accessible  
                            - Includes genuine reactions and natural speech patterns
                            - Concludes with key takeaways woven naturally into closing dialogue
                            - Follows exact formatting: Each dialogue line separate with "{host1}:" or "{host2}:" tags

                            Aim for a substantial, in-depth discussion (2000+ words) that thoroughly explores the summarized material while keeping listeners engaged and entertained."""
        }
    ]


def convert_script_format(podcast_script: str, alex_to_speaker: str = "[MIKE]", romen_to_speaker: str = "[JANE]") -> str:
    lines = podcast_script.strip().split('\n')
    converted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Convert Alex to specified speaker
        if line.startswith('Alex:'):
            converted_line = line.replace('Alex:', alex_to_speaker, 1)
            converted_lines.append(converted_line)
        # Convert Romen to specified speaker  
        elif line.startswith('Romen:'):
            converted_line = line.replace('Romen:', romen_to_speaker, 1)
            converted_lines.append(converted_line)
        # Skip other lines (like scratchpad content)
        
    return '\n'.join(converted_lines)



def clean_podcast_script(script_text: str) -> list:
    """
    Clean and parse the podcast script into individual dialogue lines
    """
    lines = []
    
    # Split by lines and clean
    raw_lines = script_text.strip().split('\n')
    
    for line in raw_lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip stage directions and music cues
        if line.startswith('**[') and line.endswith(']**'):
            continue
            
        # Skip markdown formatting lines
        if line.startswith('---') or line == '**[END OF EPISODE]**':
            continue
            
        # Skip scratchpad content
        if line.startswith('<scratchpad>') or line.startswith('</scratchpad>'):
            continue
            
        # Only keep dialogue lines
        if line.startswith('[Alex]') or line.startswith('[Romen]'):
            lines.append(line)
    
    return lines


def clean_utterance_for_tts(utterance: str) -> str:
    """
    Clean utterance text for optimal TTS processing
    """
    # Remove markdown formatting
    utterance = re.sub(r'\*\*(.*?)\*\*', r'\1', utterance)  # Remove **bold**
    utterance = re.sub(r'\*(.*?)\*', r'\1', utterance)      # Remove *italic*
    
    # Clean up special characters that might cause TTS issues
    utterance = utterance.replace('—', ' - ')  # Em dash to regular dash
    utterance = utterance.replace('"', '"').replace('"', '"')  # Smart quotes
    utterance = utterance.replace(''', "'").replace(''', "'")  # Smart apostrophes
    
    # Handle acronyms for better pronunciation
    acronym_replacements = {
        'AI': 'A I',
        'API': 'A P I',
        'TTS': 'T T S',
        'GPT': 'G P T',
        'MoE': 'M o E',
        'AIME': 'A I M E',
        'Qwen3': 'Q-wen 3',
        'GPT-4o': 'G P T 4 o',
        'o3': 'o 3'
    }
    
    for acronym, replacement in acronym_replacements.items():
        utterance = re.sub(rf'\b{acronym}\b', replacement, utterance)
    
    # Remove extra whitespace
    utterance = ' '.join(utterance.split())
    
    return utterance


def extract_pdf_content(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        
        reader = PdfReader(pdf_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        print(f"Failed to extract PDF: {e}")
        return ""
