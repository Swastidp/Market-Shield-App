import re
from typing import Dict, List, Any
import numpy as np

class RiskAnalyzer:
    """Advanced risk analysis for fraud detection."""
    
    def __init__(self):
        self.fraud_patterns = self._load_fraud_patterns()
        self.weights = {
            'impersonation': 0.30,
            'guarantee_claims': 0.25,
            'social_manipulation': 0.20,
            'registry_mismatch': 0.15,
            'artifact_tampering': 0.10
        }
    
    def _load_fraud_patterns(self) -> Dict[str, List[str]]:
        """Load fraud detection patterns."""
        return {
            'guaranteed_returns': [
                r'100% profit', r'guaranteed gains?', r'sure shot returns?',
                r'risk[- ]?free', r'no loss', r'assured returns?',
                r'guaranteed (?:profit|return|income)', r'fixed returns?'
            ],
            'impersonation_claims': [
                r'sebi registered', r'rbi approved', r'government certified',
                r'licensed advisor', r'certified analyst', r'authorized dealer',
                r'sebi (?:registration|license) no[.:]\s*\w+'
            ],
            'urgency_manipulation': [
                r'act (?:fast|now|quickly)', r'limited time', r'urgent',
                r'today only', r'expires (?:today|soon)', r'hurry up',
                r'don\'t miss', r'last chance', r'immediate action'
            ],
            'social_proof_manipulation': [
                r'exclusive (?:tip|offer)', r'vip (?:group|members?)',
                r'secret (?:formula|strategy)', r'insider (?:info|tip)',
                r'only for selected', r'premium members?'
            ],
            'payment_red_flags': [
                r'send money', r'transfer amount', r'pay (?:now|first)',
                r'advance payment', r'registration fee', r'joining fee',
                r'upi id[:\s]*[\w@.-]+', r'paytm number'
            ]
        }
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Comprehensive fraud risk analysis."""
        content_lower = content.lower()
        
        # Calculate individual risk scores
        scores = {}
        detected_features = []
        
        # 1. Impersonation detection
        impersonation_score = self._detect_impersonation(content_lower)
        scores['impersonation'] = impersonation_score
        if impersonation_score > 0.3:
            detected_features.append("Potential impersonation of registered entities")
        
        # 2. Guaranteed returns claims
        guarantee_score = self._detect_guarantees(content_lower)
        scores['guarantee_claims'] = guarantee_score
        if guarantee_score > 0.4:
            detected_features.append("Guaranteed returns or risk-free claims detected")
        
        # 3. Social manipulation tactics
        manipulation_score = self._detect_manipulation(content_lower)
        scores['social_manipulation'] = manipulation_score
        if manipulation_score > 0.3:
            detected_features.append("Urgency or social pressure tactics identified")
        
        # 4. Registry mismatch (placeholder - would need real verification)
        registry_score = self._detect_registry_issues(content_lower)
        scores['registry_mismatch'] = registry_score
        if registry_score > 0.2:
            detected_features.append("Potential registration credential mismatch")
        
        # 5. Artifact tampering indicators
        tampering_score = self._detect_tampering_indicators(content_lower)
        scores['artifact_tampering'] = tampering_score
        if tampering_score > 0.2:
            detected_features.append("Possible content manipulation indicators")
        
        # Calculate weighted final score
        final_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # Determine risk level
        if final_score >= 0.6:
            risk_level = "HIGH_RISK"
        elif final_score >= 0.3:
            risk_level = "UNCERTAIN"
        else:
            risk_level = "LEGITIMATE"
        
        # Generate reasons
        reasons = self._generate_reasons(detected_features, scores)
        
        return {
            'risk_score': final_score,
            'risk_level': risk_level,
            'component_scores': scores,
            'reasons': reasons,
            'detected_features': detected_features
        }
    
    def _detect_impersonation(self, content: str) -> float:
        """Detect impersonation attempts."""
        score = 0.0
        patterns = self.fraud_patterns['impersonation_claims']
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.3, 0.8)
        
        return min(score, 1.0)
    
    def _detect_guarantees(self, content: str) -> float:
        """Detect guaranteed returns claims."""
        score = 0.0
        patterns = self.fraud_patterns['guaranteed_returns']
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.4, 0.9)
        
        return min(score, 1.0)
    
    def _detect_manipulation(self, content: str) -> float:
        """Detect social manipulation tactics."""
        score = 0.0
        
        urgency_patterns = self.fraud_patterns['urgency_manipulation']
        social_patterns = self.fraud_patterns['social_proof_manipulation']
        
        for pattern in urgency_patterns + social_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            score += min(matches * 0.25, 0.7)
        
        return min(score, 1.0)
    
    def _detect_registry_issues(self, content: str) -> float:
        """Detect potential registry/credential issues."""
        score = 0.0
        
        # Look for SEBI registration number patterns
        sebi_pattern = r'sebi.*?(?:reg|registration|license).*?no[.:]\s*([a-z0-9/-]+)'
        matches = re.findall(sebi_pattern, content, re.IGNORECASE)
        
        if matches:
            # In production, verify against actual SEBI database
            # For demo, flag suspicious patterns
            for match in matches:
                if len(match) < 6 or not re.search(r'\d', match):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_tampering_indicators(self, content: str) -> float:
        """Detect potential content tampering."""
        score = 0.0
        
        # Look for suspicious formatting or inconsistencies
        if re.search(r'(?:copy|paste|forward)', content, re.IGNORECASE):
            score += 0.1
        
        # Check for unusual character patterns
        if len(re.findall(r'[^\w\s.,!?-]', content)) > len(content) * 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_reasons(self, features: List[str], scores: Dict[str, float]) -> List[str]:
        """Generate human-readable reasons for the risk assessment."""
        reasons = []
        
        # Add detected features as reasons
        reasons.extend(features[:3])
        
        # Add score-based reasons if no features detected
        if not reasons:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            for component, score in sorted_scores[:3]:
                if score > 0.1:
                    reason_map = {
                        'impersonation': 'Content may contain unverified credential claims',
                        'guarantee_claims': 'Presence of suspicious return guarantees',
                        'social_manipulation': 'Social pressure or urgency tactics detected',
                        'registry_mismatch': 'Potential regulatory compliance issues',
                        'artifact_tampering': 'Content authenticity concerns identified'
                    }
                    reasons.append(reason_map.get(component, f'Risk factor: {component}'))
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append("Content appears legitimate based on current analysis")
        
        return reasons[:3]
