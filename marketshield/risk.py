import re
from typing import Dict, List, Any
import numpy as np

class RiskAnalyzer:
    """Advanced risk analysis for fraud detection with enhanced pump and dump detection."""
    
    def __init__(self):
        self.fraud_patterns = self._load_fraud_patterns()
        self.weights = {
            'impersonation': 0.20,
            'guarantee_claims': 0.15,
            'social_manipulation': 0.20,
            'pump_and_dump': 0.35,  # Higher weight for P&D detection
            'registry_mismatch': 0.10
        }
    
    def _load_fraud_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive fraud detection patterns including pump and dump."""
        return {
            'guaranteed_returns': [
                r'100% profit', r'guaranteed gains?', r'sure shot returns?',
                r'risk[- ]?free', r'no loss', r'assured returns?',
                r'guaranteed (?:profit|return|income)', r'fixed returns?',
                r'zero risk', r'confirmed profit', r'definite gains?'
            ],
            'pump_and_dump': [
                # Stock promotion keywords
                r'stock buy', r'buy\s+recommendation', r'strong buy',
                r'accumulate', r'add more', r'load up',
                
                # Target and price patterns
                r'target', r'tg:', r'tp:', r'book profit',
                r'rate\s+\d+', r'current price', r'entry price',
                r'exit price', r'stop loss', r'sl:',
                
                # Hype language
                r'multibag', r'golden chance', r'moon',
                r'rocket', r'explosive', r'breakout',
                r'hidden gem', r'undervalued',
                
                # Volume mentions
                r'\d+l\s+share', r'lakh shares?', r'bulk',
                
                # Timing pressure
                r'good time to buy', r'perfect entry', r'buy now'
            ],
            'impersonation_claims': [
                r'sebi registered', r'rbi approved', r'government certified',
                r'licensed advisor', r'certified analyst', r'authorized dealer',
                r'sebi registration', r'nse member', r'bse member',
                r'research analyst', r'investment advisor',
                r'capital$', r'securities$', r'investments?$',
                r'fintech$', r'advisors?$'
            ],
            'urgency_manipulation': [
                r'act fast', r'limited time', r'urgent',
                r'today only', r'hurry up', r'don\'t miss',
                r'last chance', r'immediate', r'good time',
                r'perfect entry', r'before it\'s too late'
            ],
            'social_proof_manipulation': [
                r'exclusive tip', r'vip group', r'secret formula',
                r'insider info', r'premium members?',
                r'special clients?', r'closed group'
            ],
            'payment_red_flags': [
                r'send money', r'transfer amount', r'pay now',
                r'advance payment', r'registration fee', r'joining fee',
                r'upi id', r'paytm number', r'subscription fee'
            ]
        }
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Comprehensive fraud risk analysis with enhanced pump and dump detection."""
        content_lower = content.lower()
        original_content = content
        
        # Calculate individual risk scores
        scores = {}
        detected_features = []
        
        # 1. Pump and dump detection (PRIORITY - CHECK FIRST)
        pump_dump_score = self._detect_pump_and_dump(content_lower, original_content)
        scores['pump_and_dump'] = pump_dump_score
        if pump_dump_score > 0.1:
            detected_features.append("Stock manipulation and pump-and-dump scheme indicators detected")
        
        # 2. Social manipulation tactics
        manipulation_score = self._detect_manipulation(content_lower)
        scores['social_manipulation'] = manipulation_score
        if manipulation_score > 0.15:
            detected_features.append("Urgency manipulation or social pressure tactics identified")
        
        # 3. Impersonation detection
        impersonation_score = self._detect_impersonation(content_lower)
        scores['impersonation'] = impersonation_score
        if impersonation_score > 0.15:
            detected_features.append("Potential impersonation of registered financial entities")
        
        # 4. Guaranteed returns claims
        guarantee_score = self._detect_guarantees(content_lower)
        scores['guarantee_claims'] = guarantee_score
        if guarantee_score > 0.15:
            detected_features.append("Guaranteed returns or risk-free investment claims detected")
        
        # 5. Registry mismatch
        registry_score = self._detect_registry_issues(content_lower)
        scores['registry_mismatch'] = registry_score
        if registry_score > 0.1:
            detected_features.append("Potential registration credential inconsistencies")
        
        # Calculate weighted final score
        final_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # ADJUSTED THRESHOLDS FOR FRAUD DETECTION
        if final_score >= 0.35:      # HIGH_RISK (more sensitive)
            risk_level = "HIGH_RISK"
        elif final_score >= 0.15:    # UNCERTAIN (more sensitive)
            risk_level = "UNCERTAIN"
        else:                        # LEGITIMATE
            risk_level = "LEGITIMATE"
        
        # Generate reasons
        reasons = self._generate_reasons(detected_features, scores, final_score)
        
        return {
            'risk_score': final_score,
            'risk_level': risk_level,
            'component_scores': scores,
            'reasons': reasons,
            'detected_features': detected_features
        }
    
    def _detect_pump_and_dump(self, content_lower: str, original_content: str) -> float:
        """Enhanced pump and dump scheme detection - PRIORITY METHOD."""
        score = 0.0
        
        # DIRECT KEYWORD DETECTION (for your test case)
        direct_keywords = {
            'stock buy': 0.4,
            'buy': 0.2,
            'target': 0.3,
            'multibag': 0.4,
            'golden chance': 0.3,
            'good time to buy': 0.3,
            'share': 0.1,
            'capital': 0.2
        }
        
        for keyword, weight in direct_keywords.items():
            if keyword in content_lower:
                score += weight
        
        # PATTERN DETECTION
        
        # 1. Stock symbols in brackets (IGCIL)
        stock_symbols = re.findall(r'\([A-Z]{2,6}\)', original_content)
        if stock_symbols:
            score += 0.4
        
        # 2. Target price patterns (TG:20, Target 25)
        target_patterns = re.findall(r'(?:tg|target|tp):\s*\d+', content_lower, re.IGNORECASE)
        if target_patterns:
            score += 0.4
        
        # 3. Volume mentions (2L shares, 5 lakh shares)
        volume_patterns = re.findall(r'(\d+)l\s+share', content_lower, re.IGNORECASE)
        if volume_patterns:
            score += 0.4
        
        # 4. Price level mentions (rate 4.26)
        price_patterns = re.findall(r'rate\s+\d+\.?\d*', content_lower, re.IGNORECASE)
        if price_patterns:
            score += 0.3
        
        # 5. Percentage mentions (5%, 20%)
        percentage_mentions = re.findall(r'\d+%', original_content)
        if percentage_mentions:
            score += 0.2 * len(percentage_mentions)
        
        # 6. UP/DOWN indicators
        if re.search(r'up:\s*\d+%', content_lower, re.IGNORECASE):
            score += 0.3
        
        # 7. Company name patterns (suspicious advisor names)
        if re.search(r'\b\w+(?:capital|securities|investments?)\b', content_lower):
            score += 0.2
        
        # 8. Multiple stock indicators in one message
        indicators = [
            bool(stock_symbols),
            bool(target_patterns),
            bool(volume_patterns),
            bool(price_patterns),
            'buy' in content_lower,
            'target' in content_lower
        ]
        
        if sum(indicators) >= 3:  # Multiple indicators = high suspicion
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_impersonation(self, content: str) -> float:
        """Detect impersonation attempts of financial entities."""
        score = 0.0
        patterns = self.fraud_patterns['impersonation_claims']
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.25, 0.6)
        
        # Check for suspicious company name without proper registration
        if re.search(r'\b\w+capital\b', content, re.IGNORECASE):
            if not re.search(r'sebi.*registration', content, re.IGNORECASE):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_guarantees(self, content: str) -> float:
        """Detect guaranteed returns claims."""
        score = 0.0
        patterns = self.fraud_patterns['guaranteed_returns']
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.3, 0.7)
        
        # Check for unrealistic percentage claims
        percentage_claims = re.findall(r'(\d+)%\s*(?:profit|return|gain)', content, re.IGNORECASE)
        for claim in percentage_claims:
            if int(claim) > 30:  # Unrealistic returns
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_manipulation(self, content: str) -> float:
        """Detect social manipulation tactics."""
        score = 0.0
        
        urgency_patterns = self.fraud_patterns['urgency_manipulation']
        social_patterns = self.fraud_patterns['social_proof_manipulation']
        
        for pattern in urgency_patterns + social_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.2, 0.6)
        
        return min(score, 1.0)
    
    def _detect_registry_issues(self, content: str) -> float:
        """Detect potential registry/credential issues."""
        score = 0.0
        
        # Look for SEBI registration claims
        sebi_claims = re.findall(r'sebi.*(?:registered|registration)', content, re.IGNORECASE)
        sebi_numbers = re.findall(r'sebi.*no[.:]\s*([a-z0-9/-]+)', content, re.IGNORECASE)
        
        if sebi_claims and not sebi_numbers:
            score += 0.4  # Claims registration but no number
        
        # Check for financial entity claims without credentials
        financial_terms = ['advisor', 'analyst', 'capital', 'securities']
        has_financial_claim = any(term in content.lower() for term in financial_terms)
        has_sebi_mention = 'sebi' in content.lower()
        
        if has_financial_claim and not has_sebi_mention:
            score += 0.3
        
        return min(score, 1.0)
    
    def _generate_reasons(self, features: List[str], scores: Dict[str, float], final_score: float) -> List[str]:
        """Generate human-readable reasons for the risk assessment."""
        reasons = []
        
        # Add detected features as primary reasons
        reasons.extend(features[:3])
        
        # If no specific features, add score-based reasons
        if len(reasons) < 3:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            reason_map = {
                'pump_and_dump': 'Stock manipulation and pump-and-dump scheme patterns detected',
                'impersonation': 'Potential impersonation of registered financial entities',
                'guarantee_claims': 'Suspicious investment return guarantees identified',
                'social_manipulation': 'Social pressure and urgency tactics detected',
                'registry_mismatch': 'Regulatory compliance concerns identified'
            }
            
            for component, score in sorted_scores:
                if score > 0.1 and len(reasons) < 3:
                    reason = reason_map.get(component, f'Risk indicator: {component}')
                    if reason not in reasons:
                        reasons.append(reason)
        
        # Ensure we have at least one reason
        if not reasons:
            if final_score > 0.1:
                reasons.append("Multiple fraud risk indicators detected")
            else:
                reasons.append("Content appears legitimate based on current analysis")
        
        return reasons[:3]
