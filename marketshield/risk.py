import re
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import streamlit as st

# Import caching decorator
from .cache import cached_analysis

class RiskAnalyzer:
    """
    Advanced risk analysis for fraud detection with enhanced pump and dump detection,
    performance caching, and comprehensive pattern recognition.
    """

    def __init__(self):
        self.fraud_patterns = self._load_fraud_patterns()
        self.weights = {
            'impersonation': 0.20,
            'guarantee_claims': 0.15,
            'social_manipulation': 0.20,
            'pump_and_dump': 0.35,  # Higher weight for P&D detection
            'registry_mismatch': 0.10
        }
        
        # Performance tracking
        self.analysis_count = 0
        self.cache_hits = 0
        
        # Enhanced pattern weights for better detection
        self.enhanced_weights = {
            'financial_keywords': {
                'stock': 0.3, 'share': 0.3, 'investment': 0.2,
                'profit': 0.2, 'returns': 0.2, 'trading': 0.3
            },
            'urgency_indicators': {
                'urgent': 0.4, 'immediate': 0.3, 'limited time': 0.5,
                'act fast': 0.5, 'don\'t miss': 0.4
            },
            'credibility_claims': {
                'sebi registered': 0.6, 'government approved': 0.5,
                'licensed': 0.4, 'certified': 0.3
            }
        }

    def _load_fraud_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive fraud detection patterns with enhanced coverage."""
        return {
            'guaranteed_returns': [
                r'100% profit', r'guaranteed gains?', r'sure shot returns?',
                r'risk[- ]?free', r'no loss', r'assured returns?',
                r'guaranteed (?:profit|return|income)', r'fixed returns?',
                r'zero risk', r'confirmed profit', r'definite gains?',
                r'no risk investment', r'safe investment', r'fixed profit'
            ],
            'pump_and_dump': [
                # Stock promotion keywords
                r'stock buy', r'buy\s+recommendation', r'strong buy',
                r'accumulate', r'add more', r'load up', r'collect',
                # Target and price patterns
                r'target', r'tg:', r'tp:', r'book profit', r'profit booking',
                r'rate\s+\d+', r'current price', r'entry price', r'cmp',
                r'exit price', r'stop loss', r'sl:', r'stoploss',
                # Hype language
                r'multibag', r'golden chance', r'moon', r'to the moon',
                r'rocket', r'explosive', r'breakout', r'rally',
                r'hidden gem', r'undervalued', r'gem stock',
                # Volume mentions
                r'\d+l\s+share', r'lakh shares?', r'bulk', r'heavy volume',
                # Timing pressure
                r'good time to buy', r'perfect entry', r'buy now',
                r'last chance', r'before it flies'
            ],
            'impersonation_claims': [
                r'sebi registered', r'rbi approved', r'government certified',
                r'licensed advisor', r'certified analyst', r'authorized dealer',
                r'sebi registration', r'nse member', r'bse member',
                r'research analyst', r'investment advisor', r'portfolio manager',
                r'capital$', r'securities$', r'investments?$',
                r'fintech$', r'advisors?$', r'wealth management'
            ],
            'urgency_manipulation': [
                r'act fast', r'limited time', r'urgent', r'hurry',
                r'today only', r'hurry up', r'don\'t miss',
                r'last chance', r'immediate', r'good time',
                r'perfect entry', r'before it\'s too late',
                r'expires soon', r'limited slots', r'closing soon'
            ],
            'social_proof_manipulation': [
                r'exclusive tip', r'vip group', r'secret formula',
                r'insider info', r'premium members?', r'inner circle',
                r'special clients?', r'closed group', r'private group',
                r'elite members', r'selected few', r'handpicked'
            ],
            'payment_red_flags': [
                r'send money', r'transfer amount', r'pay now',
                r'advance payment', r'registration fee', r'joining fee',
                r'upi id', r'paytm number', r'subscription fee',
                r'gpay', r'phonepe', r'membership fee'
            ],
            'cryptocurrency_scams': [
                r'bitcoin', r'crypto', r'blockchain investment',
                r'nft', r'defi', r'yield farming', r'liquidity mining',
                r'crypto trading', r'digital currency'
            ]
        }

    @cached_analysis("risk_analysis", ttl=1800)  # Cache for 30 minutes
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Comprehensive fraud risk analysis with enhanced detection algorithms.
        Results are cached for 30 minutes to improve performance.
        """
        start_time = time.time()
        self.analysis_count += 1
        
        content_lower = content.lower()
        original_content = content

        # Pre-processing: Extract key indicators
        financial_indicators = self._extract_financial_indicators(content_lower)
        
        # Calculate individual risk scores
        scores = {}
        detected_features = []
        confidence_factors = []

        # 1. Pump and dump detection (PRIORITY - highest weight)
        pump_dump_score, pump_features = self._detect_pump_and_dump_enhanced(
            content_lower, original_content, financial_indicators
        )
        scores['pump_and_dump'] = pump_dump_score
        if pump_dump_score > 0.1:
            detected_features.extend(pump_features)
            confidence_factors.append(('pump_and_dump', pump_dump_score))

        # 2. Social manipulation tactics
        manipulation_score, manipulation_features = self._detect_manipulation_enhanced(content_lower)
        scores['social_manipulation'] = manipulation_score
        if manipulation_score > 0.15:
            detected_features.extend(manipulation_features)
            confidence_factors.append(('social_manipulation', manipulation_score))

        # 3. Impersonation detection
        impersonation_score, impersonation_features = self._detect_impersonation_enhanced(content_lower)
        scores['impersonation'] = impersonation_score
        if impersonation_score > 0.15:
            detected_features.extend(impersonation_features)
            confidence_factors.append(('impersonation', impersonation_score))

        # 4. Guaranteed returns claims
        guarantee_score, guarantee_features = self._detect_guarantees_enhanced(content_lower)
        scores['guarantee_claims'] = guarantee_score
        if guarantee_score > 0.15:
            detected_features.extend(guarantee_features)
            confidence_factors.append(('guarantee_claims', guarantee_score))

        # 5. Registry mismatch
        registry_score, registry_features = self._detect_registry_issues_enhanced(content_lower)
        scores['registry_mismatch'] = registry_score
        if registry_score > 0.1:
            detected_features.extend(registry_features)
            confidence_factors.append(('registry_mismatch', registry_score))

        # 6. NEW: Cryptocurrency scam detection
        crypto_score = self._detect_crypto_scams(content_lower)
        scores['crypto_scams'] = crypto_score
        if crypto_score > 0.2:
            detected_features.append("Potential cryptocurrency investment scam indicators")

        # Calculate weighted final score with enhanced algorithm
        base_score = sum(scores[key] * self.weights.get(key, 0.1) for key in scores)
        
        # Apply confidence multiplier based on multiple indicators
        confidence_multiplier = min(1.0 + (len(confidence_factors) * 0.1), 1.5)
        final_score = min(base_score * confidence_multiplier, 1.0)

        # Enhanced risk level determination with dynamic thresholds
        risk_level, confidence_level = self._determine_risk_level(final_score, confidence_factors)

        # Generate comprehensive reasons
        reasons = self._generate_enhanced_reasons(detected_features, scores, final_score)

        # Performance metrics
        analysis_time = time.time() - start_time
        
        return {
            'risk_score': final_score,
            'risk_level': risk_level,
            'confidence_level': confidence_level,
            'component_scores': scores,
            'reasons': reasons,
            'detected_features': list(set(detected_features)),  # Remove duplicates
            'financial_indicators': financial_indicators,
            'analysis_metadata': {
                'analysis_time': analysis_time,
                'pattern_matches': sum(len(features) for features in [
                    pump_features, manipulation_features, impersonation_features,
                    guarantee_features, registry_features
                ]),
                'confidence_factors': confidence_factors
            }
        }

    def _extract_financial_indicators(self, content: str) -> Dict[str, Any]:
        """Extract financial indicators from content for enhanced analysis."""
        indicators = {
            'stock_symbols': re.findall(r'\b[A-Z]{2,6}\b|\([A-Z]{2,6}\)', content.upper()),
            'price_mentions': re.findall(r'₹?\s*\d+(?:\.\d+)?', content),
            'percentage_mentions': re.findall(r'\d+(?:\.\d+)?%', content),
            'volume_mentions': re.findall(r'\d+\s*(?:lakh|crore|l|cr)', content, re.IGNORECASE),
            'time_frames': re.findall(r'\d+\s*(?:days?|weeks?|months?|years?)', content, re.IGNORECASE),
            'financial_terms': []
        }
        
        # Count financial terms
        for term_category, terms in self.enhanced_weights.items():
            for term in terms:
                if term in content:
                    indicators['financial_terms'].append(term)
        
        return indicators

    def _detect_pump_and_dump_enhanced(self, content_lower: str, original_content: str, 
                                     financial_indicators: Dict) -> Tuple[float, List[str]]:
        """Enhanced pump and dump scheme detection with feature tracking."""
        score = 0.0
        features = []
        
        # Direct keyword detection with weighted scoring
        direct_keywords = {
            'stock buy': 0.5, 'buy recommendation': 0.6, 'strong buy': 0.7,
            'buy': 0.2, 'target': 0.4, 'multibag': 0.6,
            'golden chance': 0.5, 'good time to buy': 0.5,
            'share': 0.1, 'capital': 0.2, 'breakout': 0.4
        }

        for keyword, weight in direct_keywords.items():
            if keyword in content_lower:
                score += weight
                features.append(f"Pump-and-dump keyword detected: '{keyword}'")

        # Enhanced pattern detection
        # 1. Stock symbols analysis
        stock_symbols = financial_indicators['stock_symbols']
        if stock_symbols:
            score += min(len(stock_symbols) * 0.3, 0.6)
            features.append(f"Stock symbols detected: {', '.join(stock_symbols[:3])}")

        # 2. Target price patterns
        target_patterns = re.findall(r'(?:tg|target|tp):\s*\d+', content_lower, re.IGNORECASE)
        if target_patterns:
            score += min(len(target_patterns) * 0.4, 0.8)
            features.append("Target price mentions detected")

        # 3. Volume analysis
        volume_patterns = financial_indicators['volume_mentions']
        if volume_patterns:
            score += min(len(volume_patterns) * 0.3, 0.6)
            features.append("Volume-based trading mentions")

        # 4. Price level mentions
        price_patterns = re.findall(r'(?:rate|price|cmp)\s*:?\s*\d+\.?\d*', content_lower, re.IGNORECASE)
        if price_patterns:
            score += min(len(price_patterns) * 0.2, 0.4)
            features.append("Specific price level mentions")

        # 5. Percentage promises analysis
        percentage_mentions = financial_indicators['percentage_mentions']
        if percentage_mentions:
            high_percentages = [p for p in percentage_mentions if float(p.replace('%', '')) > 50]
            if high_percentages:
                score += min(len(high_percentages) * 0.3, 0.6)
                features.append("High percentage return promises")

        # 6. Urgency + Stock combination
        urgency_terms = ['urgent', 'immediate', 'today', 'now', 'hurry']
        stock_terms = ['stock', 'share', 'buy', 'invest']
        
        urgency_count = sum(1 for term in urgency_terms if term in content_lower)
        stock_count = sum(1 for term in stock_terms if term in content_lower)
        
        if urgency_count >= 1 and stock_count >= 1:
            score += 0.4
            features.append("Urgency combined with stock promotion")

        # 7. Multiple indicator correlation
        indicator_types = [
            bool(stock_symbols),
            bool(target_patterns),
            bool(volume_patterns),
            bool(price_patterns),
            'buy' in content_lower,
            'target' in content_lower
        ]

        if sum(indicator_types) >= 3:
            score += 0.3
            features.append("Multiple pump-and-dump indicators present")

        return min(score, 1.0), features

    def _detect_impersonation_enhanced(self, content: str) -> Tuple[float, List[str]]:
        """Enhanced impersonation detection with feature tracking."""
        score = 0.0
        features = []
        
        # Check for regulatory claims
        patterns = self.fraud_patterns['impersonation_claims']
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                score += min(len(matches) * 0.25, 0.6)
                features.append(f"Regulatory impersonation claim: {pattern}")

        # Enhanced company name analysis
        company_patterns = [
            r'\b\w+\s*(?:capital|securities|investments?|fintech|advisors?)\b',
            r'\b(?:abc|xyz|reliable|trusted|premium)\s*(?:capital|securities)\b'
        ]
        
        for pattern in company_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Check if proper registration is mentioned
                if not re.search(r'sebi.*(?:registration|no)', content, re.IGNORECASE):
                    score += 0.4
                    features.append("Financial entity name without proper credentials")

        # Check for fake credentials
        fake_credentials = [
            r'certified by sebi', r'approved by rbi', r'government licensed',
            r'iso certified', r'award winning'
        ]
        
        for credential in fake_credentials:
            if re.search(credential, content, re.IGNORECASE):
                score += 0.3
                features.append(f"Suspicious credential claim: {credential}")

        return min(score, 1.0), features

    def _detect_guarantees_enhanced(self, content: str) -> Tuple[float, List[str]]:
        """Enhanced guaranteed returns detection."""
        score = 0.0
        features = []
        
        patterns = self.fraud_patterns['guaranteed_returns']
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                score += min(len(matches) * 0.3, 0.7)
                features.append(f"Guaranteed return claim: {pattern}")

        # Analyze unrealistic percentage claims
        percentage_claims = re.findall(r'(\d+(?:\.\d+)?)%\s*(?:profit|return|gain|growth)', content, re.IGNORECASE)
        for claim in percentage_claims:
            percentage = float(claim)
            if percentage > 100:
                score += 0.8
                features.append(f"Extremely unrealistic return promise: {percentage}%")
            elif percentage > 50:
                score += 0.5
                features.append(f"Highly unrealistic return promise: {percentage}%")
            elif percentage > 30:
                score += 0.3
                features.append(f"Unrealistic return promise: {percentage}%")

        # Check for time-bound guarantees
        time_guarantees = re.findall(
            r'(?:guaranteed|assured|confirmed)\s+(?:profit|return)\s+(?:in|within)\s+\d+\s+(?:days?|weeks?|months?)', 
            content, re.IGNORECASE
        )
        if time_guarantees:
            score += 0.5
            features.append("Time-bound guaranteed returns")

        return min(score, 1.0), features

    def _detect_manipulation_enhanced(self, content: str) -> Tuple[float, List[str]]:
        """Enhanced social manipulation detection."""
        score = 0.0
        features = []
        
        urgency_patterns = self.fraud_patterns['urgency_manipulation']
        social_patterns = self.fraud_patterns['social_proof_manipulation']
        
        for pattern in urgency_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                score += min(len(matches) * 0.25, 0.6)
                features.append(f"Urgency manipulation: {pattern}")

        for pattern in social_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                score += min(len(matches) * 0.3, 0.7)
                features.append(f"Social proof manipulation: {pattern}")

        # Check for FOMO (Fear of Missing Out) tactics
        fomo_patterns = [
            r'don\'t miss out', r'limited offer', r'few spots left',
            r'closing soon', r'expires today', r'last chance'
        ]
        
        for pattern in fomo_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.4
                features.append(f"FOMO tactic detected: {pattern}")

        return min(score, 1.0), features

    def _detect_registry_issues_enhanced(self, content: str) -> Tuple[float, List[str]]:
        """Enhanced registry/credential issues detection."""
        score = 0.0
        features = []

        # Look for SEBI registration claims without numbers
        sebi_claims = re.findall(r'sebi.*(?:registered|registration)', content, re.IGNORECASE)
        sebi_numbers = re.findall(r'(?:sebi|registration)\s*no[.:\s]*([a-z0-9/-]+)', content, re.IGNORECASE)

        if sebi_claims and not sebi_numbers:
            score += 0.5
            features.append("Claims SEBI registration without providing number")

        # Check for vague regulatory claims
        vague_claims = [
            r'government approved', r'officially recognized',
            r'regulatory compliant', r'fully licensed'
        ]
        
        for claim in vague_claims:
            if re.search(claim, content, re.IGNORECASE):
                score += 0.3
                features.append(f"Vague regulatory claim: {claim}")

        # Financial entity claims without proper backing
        financial_terms = ['advisor', 'analyst', 'capital', 'securities', 'portfolio manager']
        has_financial_claim = any(term in content.lower() for term in financial_terms)
        has_specific_registration = bool(sebi_numbers) or 'registration number' in content.lower()

        if has_financial_claim and not has_specific_registration:
            score += 0.4
            features.append("Financial service claims without specific credentials")

        return min(score, 1.0), features

    def _detect_crypto_scams(self, content: str) -> float:
        """Detect cryptocurrency-related scam patterns."""
        score = 0.0
        crypto_patterns = self.fraud_patterns['cryptocurrency_scams']
        
        for pattern in crypto_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.3
        
        # Check for high-yield crypto promises
        if re.search(r'crypto.*(?:guarantee|assured|fixed).*return', content, re.IGNORECASE):
            score += 0.6
            
        return min(score, 1.0)

    def _determine_risk_level(self, score: float, confidence_factors: List) -> Tuple[str, str]:
        """Determine risk level with confidence assessment."""
        
        # Base risk level determination
        if score >= 0.7:
            risk_level = "CRITICAL_RISK"
        elif score >= 0.35:
            risk_level = "HIGH_RISK"
        elif score >= 0.15:
            risk_level = "UNCERTAIN"
        else:
            risk_level = "LEGITIMATE"
        
        # Confidence level based on multiple factors
        factor_count = len(confidence_factors)
        avg_confidence = np.mean([factor[1] for factor in confidence_factors]) if confidence_factors else 0
        
        if factor_count >= 3 and avg_confidence > 0.5:
            confidence_level = "HIGH"
        elif factor_count >= 2 and avg_confidence > 0.3:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
            
        return risk_level, confidence_level

    def _generate_enhanced_reasons(self, features: List[str], scores: Dict[str, float], 
                                 final_score: float) -> List[str]:
        """Generate comprehensive reasons with prioritization."""
        reasons = []

        # Prioritize by score impact
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        reason_map = {
            'pump_and_dump': 'Stock manipulation and pump-and-dump scheme patterns detected',
            'impersonation': 'Potential impersonation of registered financial entities',
            'guarantee_claims': 'Suspicious guaranteed return promises identified',
            'social_manipulation': 'Social pressure and urgency manipulation tactics detected',
            'registry_mismatch': 'Regulatory compliance and credential concerns identified',
            'crypto_scams': 'Cryptocurrency investment scam indicators detected'
        }

        # Add top scoring components
        for component, score in sorted_scores:
            if score > 0.1 and len(reasons) < 4:
                reason = reason_map.get(component, f'Risk indicator: {component}')
                reasons.append(f"{reason} (Score: {score:.2f})")

        # Add specific feature-based reasons
        unique_features = list(set(features))[:2]  # Top 2 unique features
        for feature in unique_features:
            if len(reasons) < 5:
                reasons.append(feature)

        # Ensure we have at least one reason
        if not reasons:
            if final_score > 0.1:
                reasons.append(f"Multiple fraud risk indicators detected (Overall Score: {final_score:.2f})")
            else:
                reasons.append("Content appears legitimate based on comprehensive analysis")

        return reasons[:5]  # Return top 5 reasons

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the analyzer."""
        return {
            'total_analyses': self.analysis_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.analysis_count, 1),
            'patterns_loaded': sum(len(patterns) for patterns in self.fraud_patterns.values()),
            'detection_categories': len(self.weights)
        }

    def batch_analyze(self, contents: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple contents in batch for improved performance."""
        results = []
        
        for i, content in enumerate(contents):
            try:
                result = self.analyze_content(content)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'risk_level': 'ERROR',
                    'risk_score': 0.5
                })
        
        return results
 