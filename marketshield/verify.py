import re
from typing import Dict, List, Any
import random

class VerificationEngine:
    """Handles UPI and registry verification (simulated for MVP)."""
    
    def __init__(self):
        # Mock SEBI registry for demo
        self.mock_sebi_registry = {
            'INA000012345': {'name': 'ABC Investment Advisers', 'status': 'Active'},
            'INH000098765': {'name': 'XYZ Research Analysts', 'status': 'Active'},
            'INZ000054321': {'name': 'PQR Securities', 'status': 'Active'}
        }
        
        # Mock validated UPI handles for demo
        self.validated_upi_handles = {
            'sebi.official@paytm',
            'rbi.authorized@upi',
            'nse.verified@okaxis'
        }
    
    def verify_content(self, content: str) -> Dict[str, Any]:
        """Perform all verification checks on content."""
        results = {
            'upi_handles': self._verify_upi_handles(content),
            'registrations': self._verify_sebi_registrations(content),
            'corporate_announcements': self._verify_corporate_announcements(content)
        }
        
        return results
    
    def _verify_upi_handles(self, content: str) -> List[Dict[str, Any]]:
        """Extract and verify UPI handles."""
        # UPI handle patterns
        upi_patterns = [
            r'([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+)',
            r'upi id[:\s]*([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+)',
            r'pay to[:\s]*([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+)'
        ]
        
        found_handles = set()
        for pattern in upi_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_handles.update(matches)
        
        results = []
        for handle in found_handles:
            # Simulate SEBI Check validation (Oct 1, 2025 rollout)
            is_validated = handle.lower() in self.validated_upi_handles
            
            # Basic format validation
            is_valid_format = bool(re.match(r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+$', handle))
            
            if is_validated:
                status = "SEBI Validated ✅"
                verified = True
            elif is_valid_format:
                status = "Format Valid - Verification Pending ❓"
                verified = False
            else:
                status = "Invalid Format ⚠️"
                verified = False
            
            results.append({
                'handle': handle,
                'verified': verified,
                'status': status,
                'format_valid': is_valid_format
            })
        
        return results
    
    def _verify_sebi_registrations(self, content: str) -> List[Dict[str, Any]]:
        """Extract and verify SEBI registration claims."""
        # SEBI registration patterns
        reg_patterns = [
            r'sebi.*?(?:reg|registration|license).*?no[.:]\s*([A-Z]{3}\d{9})',
            r'(?:reg|registration|license)\s*(?:no|number)[.:]\s*([A-Z]{3}\d{9})',
            r'([A-Z]{3}\d{9})'  # Direct SEBI number format
        ]
        
        found_registrations = set()
        for pattern in reg_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_registrations.update([m.upper() for m in matches])
        
        results = []
        for reg_num in found_registrations:
            # Simulate registry lookup
            is_valid = reg_num in self.mock_sebi_registry
            
            if is_valid:
                entity_info = self.mock_sebi_registry[reg_num]
                status = f"Valid - {entity_info['name']} ({entity_info['status']})"
            else:
                # Check format validity
                if re.match(r'^[A-Z]{3}\d{9}$', reg_num):
                    status = "Number format valid but not found in registry"
                else:
                    status = "Invalid registration number format"
            
            results.append({
                'number': reg_num,
                'valid': is_valid,
                'status': status
            })
        
        return results
    
    def _verify_corporate_announcements(self, content: str) -> Dict[str, Any]:
        """Verify corporate announcement authenticity."""
        # Look for announcement indicators
        announcement_keywords = [
            'quarterly results', 'earnings', 'dividend', 'board meeting',
            'agm', 'annual report', 'merger', 'acquisition', 'ipo'
        ]
        
        is_announcement = any(
            keyword in content.lower() 
            for keyword in announcement_keywords
        )
        
        if not is_announcement:
            return {'is_announcement': False}
        
        # Simulate credibility checks
        credibility_score = random.uniform(0.3, 0.95)  # Mock score
        
        if credibility_score > 0.8:
            credibility_status = "High credibility - appears authentic"
        elif credibility_score > 0.5:
            credibility_status = "Medium credibility - requires verification"
        else:
            credibility_status = "Low credibility - potential fabrication"
        
        return {
            'is_announcement': True,
            'credibility_score': credibility_score,
            'credibility_status': credibility_status,
            'verification_needed': credibility_score < 0.8
        }
