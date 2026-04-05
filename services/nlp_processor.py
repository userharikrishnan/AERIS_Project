import re
import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from models.tokenizer import Tokenizer
from models.intent_classifier import IntentClassifier, IDX_TO_INTENT, NUM_INTENT_CLASSES

CHECKPOINT_CLASSIFIER = "checkpoints/intent_classifier.pt"
CHECKPOINT_TOKENIZER = "checkpoints/tokenizer.json"


@dataclass
class Intent:
    """Structured intent representation with reasoning-ready fields"""
    type: str
    entities: Dict[str, str]
    confidence: float = 0.0
    raw_text: str = ""
    uncertainty: float = 0.0
    alternatives: List[Dict] = None
    margin: float = 0.0
    mode: str = "ACTION"
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
    
    def __repr__(self):
        return f"Intent(type={self.type}, entities={self.entities}, confidence={self.confidence:.2f}, uncertainty={self.uncertainty:.2f})"


@dataclass
class ClassificationResult:
    """Detailed classification result"""
    intent: str
    confidence: float
    uncertainty: float
    margin: float
    alternatives: List[Dict]
    is_uncertain: bool


class NLPProcessor:
    """
    Production-grade ML-based NLP processor for AERIS
    
    Features:
    - Neural intent classification with confidence calibration
    - Rich entity extraction with context awareness
    - Context-aware processing with memory integration
    - Heuristic fallback for classifier failures
    - Comprehensive logging and diagnostics
    """
    
    # Entity extraction patterns
    APP_PATTERN = re.compile(
        r'(chrome|firefox|edge|safari|opera|brave|vscode|sublime|notepad\+\+?|'
        r'vim|emacs|terminal|cmd|powershell|spotify|discord|slack|teams|zoom|'
        r'skype|steam|outlook|word|excel|powerpoint|calculator|camera|photos|'
        r'settings|explorer|task manager|control panel)',
        re.IGNORECASE
    )
    
    URL_PATTERN = re.compile(
        r'((?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|edu|gov|io|co|ai|'
        r'dev|app|tech|blog|site|xyz|localhost)(?:/\S*)?)',
        re.IGNORECASE
    )
    
    FILE_PATTERN = re.compile(
        r'([\w-]+\.(?:txt|pdf|csv|json|xml|py|js|html|css|md|docx?|xlsx?|'
        r'pptx?|zip|tar|gz|jpg|png|mp4|mp3))',
        re.IGNORECASE
    )
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = Tokenizer()
        self.classifier: Optional[IntentClassifier] = None
        self._load_checkpoint()
        
        # Statistics for monitoring
        self.stats = {
            'total_requests': 0,
            'confident_predictions': 0,
            'fallbacks': 0,
            'errors': 0
        }
        
    def _load_checkpoint(self):
        """Load model checkpoints with error handling"""
        # Load tokenizer
        if os.path.exists(CHECKPOINT_TOKENIZER):
            try:
                self.tokenizer.load(CHECKPOINT_TOKENIZER)
                print(f"[NLPProcessor] ✓ Tokenizer loaded: {self.tokenizer.vocab_size} words")
            except Exception as e:
                print(f"[NLPProcessor] ⚠️ Tokenizer load failed: {e}")
        else:
            print("[NLPProcessor] ⚠️ No tokenizer checkpoint - using empty vocab")
        
        # Load classifier
        if os.path.exists(CHECKPOINT_CLASSIFIER):
            try:
                vocab_size = self.tokenizer.vocab_size
                
                self.classifier = IntentClassifier(
                    vocab_size=vocab_size,
                    embed_dim=128,
                    hidden_dim=256,
                    num_heads=4,
                    dropout=0.0  # No dropout for inference
                )
                
                checkpoint = torch.load(CHECKPOINT_CLASSIFIER, map_location=self.device)
                self.classifier.load_state_dict(checkpoint)
                self.classifier.eval()
                self.classifier.to(self.device)
                
                print(f"[NLPProcessor] ✓ Intent classifier loaded: {NUM_INTENT_CLASSES} classes")
                
            except Exception as e:
                print(f"[NLPProcessor] ❌ Classifier load failed: {e}")
                self.classifier = None
        else:
            print("[NLPProcessor] ⚠️ No classifier checkpoint - will use CHAT fallback")
            self.classifier = None
    
    def _infer_mode(self, intent: str, confidence: float, uncertainty: float, margin: float) -> str:
        """
        ML-based mode inference using prediction quality.
        Certain intents are always ACTION even if margin is low —
        they are handled as non-destructive bypasses in core.py.
        """
        # Direct chat prediction
        if intent == "CHAT":
            return "CHAT"

        # Some intents are non-destructive and always actionable —
        # route them as ACTION so they reach the BYPASS_ACTIONS path
        ALWAYS_ACTION = {
            "IDENTITY_QUERY", "MEMORY_RECALL", "MEMORY_STORE", "MEMORY_FORGET",
            "REASONING", "GOAL_LIST", "ACTIVE_WINDOW", "LIST_WINDOWS",
            "SYSTEM_INFO", "SCREENSHOT",
            "WEB_SCRAPE", "GENERATE_REPORT",
        }
        if intent in ALWAYS_ACTION and confidence >= 0.5:
            return "ACTION"

        # Low separation between top intents → model confused
        if margin < 0.15:
            # If the top intent was CHAT, keep it. Otherwise it's an uncertain action.
            return "CHAT" if intent == "CHAT" else "ACTION"

        # High uncertainty → don't break execution chains just because of noise
        if uncertainty > 0.5:
            return "CHAT" if intent == "CHAT" else "ACTION"

        # Weak confidence → not actionable, needs clarification, but don't force 'CHAT'
        if confidence < 0.4:
            return "CHAT" if intent == "CHAT" else "ACTION"

        return "ACTION"

    
    def extract_intent(self, text: str) -> Intent:
        """
        Main entry point: Extract intent and entities from text
        
        Args:
            text: Raw user input
            
        Returns:
            Intent object with type, entities, and confidence
        """
        self.stats['total_requests'] += 1
        
        if not text or not isinstance(text, str):
            self.stats['errors'] += 1
            return Intent(type="CHAT", entities={}, confidence=0.0, raw_text=str(text))
        
        raw_text = text
        cleaned = text.lower().strip()
        
        # Classification
        classification = self._classify(cleaned)
        
        # Entity extraction
        entities = self._extract_entities(classification.intent, cleaned, raw_text)
        
        # Update stats
        if classification.is_uncertain or classification.intent == "CHAT":
            self.stats['fallbacks'] += 1
        else:
            self.stats['confident_predictions'] += 1
        
        mode = self._infer_mode(
            classification.intent,
            classification.confidence,
            classification.uncertainty,
            classification.margin
        )
        
        # DEBUG (important for tuning)
        print(f"[NLP DEBUG] intent={classification.intent}, conf={classification.confidence:.2f}, unc={classification.uncertainty:.2f}, margin={classification.margin:.2f}, mode={mode}")
        
        return Intent(
            type=classification.intent,
            entities=entities,
            confidence=classification.confidence,
            raw_text=raw_text,
            uncertainty=classification.uncertainty,
            alternatives=classification.alternatives,
            margin=classification.margin,
            mode=mode
        )
    
    def process(self, text: str, context: dict = None) -> Intent:
        """
        Context-aware intent processing with memory integration
        
        Args:
            text: Raw user input
            context: Optional context dict containing memory and state
            
        Returns:
            Intent object with context-enhanced confidence
        """
        intent = self.extract_intent(text)
        
        if context:
            # Boost confidence if matches recent memory
            recent = context.get("memory", {}).get("recent", [])
            for mem in recent:
                if intent.raw_text in str(mem):
                    intent.confidence += 0.1
                    # Cap at 1.0
                    if intent.confidence > 1.0:
                        intent.confidence = 1.0
            
            # Penalize uncertainty if conflicting intent
            if intent.uncertainty > 0.6:
                intent.confidence *= 0.8
        
        return intent
    
    def _classify(self, text: str) -> ClassificationResult:
        """
        Neural intent classification with confidence calibration
        Includes heuristic fallback for classifier failures
        
        Args:
            text: Cleaned input text
            
        Returns:
            ClassificationResult with detailed information
        """
        if self.classifier is None:
            # Heuristic fallback when classifier is unavailable
            text_lower = text.lower()
            if "open" in text_lower:
                return ClassificationResult(
                    intent="OPEN_APP",
                    confidence=0.7,
                    uncertainty=0.3,
                    margin=0.4,
                    alternatives=[{"intent": "CHAT", "confidence": 0.3}],
                    is_uncertain=False
                )
            if "search" in text_lower:
                return ClassificationResult(
                    intent="WEB_SEARCH",
                    confidence=0.7,
                    uncertainty=0.3,
                    margin=0.4,
                    alternatives=[{"intent": "CHAT", "confidence": 0.3}],
                    is_uncertain=False
                )
            return ClassificationResult(
                intent="CHAT",
                confidence=0.0,
                uncertainty=1.0,
                margin=0.0,
                alternatives=[],
                is_uncertain=True
            )
        
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return ClassificationResult(
                intent="CHAT",
                confidence=0.0,
                uncertainty=1.0,
                margin=0.0,
                alternatives=[],
                is_uncertain=True
            )
        
        # Prepare input
        input_tensor = torch.tensor([token_ids], device=self.device)
        
        # Get detailed prediction
        with torch.no_grad():
            result = self.classifier.predict_with_confidence(input_tensor)
        
        return ClassificationResult(
            intent=result['intent'],
            confidence=result['confidence'],
            uncertainty=result['uncertainty'],
            margin=result['margin'],
            alternatives=result['alternatives'],
            is_uncertain=result['is_uncertain']
        )
    
    def _extract_entities(self, intent_type: str, text: str, raw_text: str) -> Dict[str, str]:
        """
        Context-aware entity extraction
        
        Args:
            intent_type: Predicted intent type
            text: Lowercase cleaned text
            raw_text: Original raw text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # APP_OPEN / APP_CLOSE
        if intent_type in {"OPEN_APP", "CLOSE_APP"}:
            # Try regex pattern first
            match = self.APP_PATTERN.search(raw_text)
            if match:
                entities['app'] = match.group(1).lower()
            else:
                # Fallback: remove intent keywords
                app = re.sub(
                    r'\b(open|launch|start|run|bring up|fire up|close|exit|quit|'
                    r'terminate|stop|kill|the|app|application|please|can you|'
                    r'could you|would you|for me|now|immediately)\b',
                    '',
                    text,
                    flags=re.IGNORECASE
                ).strip()
                entities['app'] = app if app else 'unknown'
        
        # WEB_SEARCH
        elif intent_type == "WEB_SEARCH":
            # Remove search keywords
            query = re.sub(
                r'\b(search the web|search online|google|look this up|look up|'
                r'web search|find me|search for|find|information about|'
                r'please|can you|could you|for me)\b',
                '',
                text,
                flags=re.IGNORECASE
            ).strip()
            entities['query'] = query if query else raw_text
        
        # WEB_NAVIGATE
        elif intent_type == "WEB_NAVIGATE":
            # Try URL pattern
            match = self.URL_PATTERN.search(raw_text)
            if match:
                url = match.group(1)
                # Add https if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                entities['url'] = url
            else:
                # Fallback: extract domain-like strings
                domain = re.sub(
                    r'\b(go to|navigate to|browse to|open|visit|website|'
                    r'please|can you|could you|for me|the)\b',
                    '',
                    text,
                    flags=re.IGNORECASE
                ).strip()
                entities['url'] = f"https://{domain}" if domain else "https://google.com"

        # WEB_SCRAPE
        elif intent_type == "WEB_SCRAPE":
            # Try URL pattern first
            match = self.URL_PATTERN.search(raw_text)
            if match:
                url = match.group(1)
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                entities['url'] = url
            else:
                entities['url'] = 'current_page'
            # Check if user wants to save results
            if any(w in text.lower() for w in ['save', 'store', 'write', 'report', 'desktop', 'file']):
                entities['save'] = True

        # GENERATE_REPORT
        elif intent_type == "GENERATE_REPORT":
            # Detect format
            fmt = 'md'  # default markdown
            if 'pdf' in text.lower():
                fmt = 'pdf'
            elif 'html' in text.lower():
                fmt = 'html'
            elif 'txt' in text.lower() or 'text' in text.lower():
                fmt = 'txt'
            entities['format'] = fmt
            # Detect save path hints
            if 'desktop' in text.lower():
                entities['save_path'] = 'desktop'
            elif 'c drive' in text.lower() or 'c:/' in text.lower():
                entities['save_path'] = 'C:/AERIS_Reports'
            elif 'download' in text.lower():
                entities['save_path'] = 'downloads'
            else:
                entities['save_path'] = 'desktop'
            # Report title hint
            title = re.sub(
                r'\b(generate|create|make|write|produce|a|the|report|document|'
                r'pdf|html|markdown|md|txt|text|summary|and|save|to|on|my)\b',
                '', text, flags=re.IGNORECASE
            ).strip()
            entities['title'] = title if len(title) > 2 else 'AERIS Report'

        # SYSTEM_INFO
        elif intent_type == "SYSTEM_INFO":
            # Detect what subsystem the user is asking about
            sub = 'full'
            if any(w in text.lower() for w in ['cpu', 'processor', 'core']):
                sub = 'cpu'
            elif any(w in text.lower() for w in ['ram', 'memory']):
                sub = 'memory'
            elif any(w in text.lower() for w in ['disk', 'storage', 'drive', 'space', 'free']):
                sub = 'disk'
            elif any(w in text.lower() for w in ['battery', 'charge', 'power', 'plugged']):
                sub = 'battery'
            elif any(w in text.lower() for w in ['ip', 'network', 'wifi', 'internet']):
                sub = 'network'
            elif any(w in text.lower() for w in ['uptime', 'how long']):
                sub = 'uptime'
            entities['subsystem'] = sub

        # SCREENSHOT
        elif intent_type == "SCREENSHOT":
            # Save path detection
            if 'desktop' in text.lower():
                entities['save_path'] = 'desktop'
            elif 'document' in text.lower():
                entities['save_path'] = 'documents'
            else:
                entities['save_path'] = 'desktop'
        
        # FILE operations
        elif intent_type in {"FILE_READ", "FILE_WRITE", "FILE_DELETE"}:
            match = self.FILE_PATTERN.search(raw_text)
            if match:
                entities['filename'] = match.group(1)
            else:
                # Fallback extraction
                filename = re.sub(
                    r'\b(create|make|new|write|read|open|show|delete|remove|'
                    r'erase|the|file|document|called|named|please|a|an)\b',
                    '',
                    text,
                    flags=re.IGNORECASE
                ).strip()
                entities['filename'] = filename if filename else 'unnamed.txt'
        
        # MEMORY_STORE
        elif intent_type == "MEMORY_STORE":
            content = re.sub(
                r'\b(remember this|save this|store this|keep this|memorize|'
                r'remember|that|please|for me|in your memory)\b',
                '',
                text,
                flags=re.IGNORECASE
            ).strip()
            entities['content'] = content if content else raw_text
        
        # MEMORY_RECALL
        elif intent_type == "MEMORY_RECALL":
            # Extract what to recall
            key = re.sub(
                r'\b(what do you remember|recall|remember|what did i say|'
                r'tell me|about|the|please)\b',
                '',
                text,
                flags=re.IGNORECASE
            ).strip()
            entities['key'] = key if key else 'general'
        
        # GOAL_CREATE
        elif intent_type == "GOAL_CREATE":
            description = re.sub(
                r'\b(set a goal|create a goal|add a goal|new goal|make a goal|'
                r'set a target|objective|to|for|please|can you|help me|'
                r'i want to|i need to|my goal is)\b',
                '',
                text,
                flags=re.IGNORECASE
            ).strip()
            entities['description'] = description if description else raw_text
        
        # REASONING
        elif intent_type == "REASONING":
            entities['topic'] = text
            entities['text'] = raw_text
        
        # IDENTITY_QUERY
        elif intent_type == "IDENTITY_QUERY":
            entities['query_type'] = 'identity'
        
        # CHAT / fallback
        elif intent_type == "CHAT":
            entities['text'] = raw_text
        
        # Add timestamp for tracking
        import time
        entities['_extracted_at'] = time.time()
        
        return entities
    
    def batch_process(self, texts: List[str]) -> List[Intent]:
        """Process multiple texts efficiently"""
        return [self.extract_intent(text) for text in texts]
    
    def get_attention_visualization(self, text: str) -> Optional[List[Dict]]:
        """Get attention weights for interpretability"""
        if self.classifier is None:
            return None
        
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return None
        
        input_tensor = torch.tensor([token_ids], device=self.device)
        return self.classifier.get_attention_visualization(input_tensor, self.tokenizer)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'confident_ratio': self.stats['confident_predictions'] / total,
            'fallback_ratio': self.stats['fallbacks'] / total,
            'error_ratio': self.stats['errors'] / total
        }
    
    def reload_model(self):
        """Reload model from checkpoint (for hot-swapping)"""
        print("[NLPProcessor] Reloading model...")
        self._load_checkpoint()
        self.stats = {k: 0 for k in self.stats}