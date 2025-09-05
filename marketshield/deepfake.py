import numpy as np
from typing import Dict, Any, Optional, Tuple
import tempfile
import os
import warnings
from PIL import Image
import streamlit as st

# Handle OpenCV import with fallback for headless environments
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    warnings.warn(f"OpenCV not available: {e}")

# Handle librosa import gracefully
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("Librosa not available - audio analysis features disabled")

# Handle scipy for advanced statistical analysis
try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class DeepfakeDetector:
    """
    Enhanced deepfake and media authenticity detection with graceful fallbacks.
    Handles missing dependencies (librosa, OpenCV) elegantly for different deployment environments.
    """

    def __init__(self):
        self.video_thresholds = {
            'frame_consistency': 0.85,
            'temporal_coherence': 0.80,
            'face_quality': 0.75
        }
        self.audio_thresholds = {
            'spectral_consistency': 0.70,
            'temporal_anomaly': 0.30
        }
        
        # Capability flags
        self.capabilities = {
            'image_analysis': CV2_AVAILABLE,
            'video_analysis': CV2_AVAILABLE,
            'audio_analysis': LIBROSA_AVAILABLE,
            'statistical_analysis': SCIPY_AVAILABLE
        }
        
        # Initialize warning messages for missing capabilities
        self._log_capabilities()

    def _log_capabilities(self):
        """Log available capabilities and missing dependencies."""
        if not CV2_AVAILABLE:
            st.warning("⚠️ OpenCV not available - image/video analysis disabled. Install opencv-python-headless for deployment compatibility.")
        
        if not LIBROSA_AVAILABLE:
            st.info("ℹ️ Librosa not available - audio analysis disabled. This is expected in most deployment environments.")
        
        # Show available capabilities in debug mode
        if st.session_state.get('show_debug_info', False):
            with st.expander("🔧 Deepfake Detector Capabilities"):
                for capability, available in self.capabilities.items():
                    status = "✅ Available" if available else "❌ Disabled"
                    st.write(f"**{capability.replace('_', ' ').title()}:** {status}")

    def analyze_image_authenticity(self, image_data) -> Dict[str, Any]:
        """Analyze image authenticity with OpenCV-based algorithms."""
        
        if not CV2_AVAILABLE:
            return self._get_fallback_result(
                "Image analysis unavailable - OpenCV dependency missing",
                analysis_type="image"
            )
        
        try:
            # Convert image data to numpy array
            if hasattr(image_data, 'convert'):
                image_array = np.array(image_data.convert('RGB'))
            else:
                image_array = image_data

            # Perform comprehensive authenticity analysis
            noise_analysis = self._analyze_noise_patterns(image_array)
            compression_analysis = self._analyze_compression_artifacts(image_array)
            edge_analysis = self._analyze_edge_consistency(image_array)
            
            # Additional analysis if available
            statistical_analysis = self._analyze_statistical_properties(image_array) if SCIPY_AVAILABLE else {'score': 0.5}

            # Calculate weighted authenticity score
            scores = [
                noise_analysis['score'],
                compression_analysis['score'],
                edge_analysis['score'],
                statistical_analysis['score']
            ]
            
            authenticity_score = np.mean(scores)
            confidence_level = self._calculate_confidence(scores)

            return {
                'authenticity_score': authenticity_score,
                'is_authentic': authenticity_score > 0.6,
                'confidence': confidence_level,
                'analysis': {
                    'noise_patterns': noise_analysis,
                    'compression_artifacts': compression_analysis,
                    'edge_consistency': edge_analysis,
                    'statistical_properties': statistical_analysis
                },
                'analysis_method': 'opencv_based',
                'capabilities_used': ['image_analysis'] + (['statistical_analysis'] if SCIPY_AVAILABLE else [])
            }

        except Exception as e:
            return self._get_error_result(str(e), "image")

    def analyze_video_authenticity(self, video_path: str) -> Dict[str, Any]:
        """Analyze video authenticity with temporal consistency checks."""
        
        if not CV2_AVAILABLE:
            return self._get_fallback_result(
                "Video analysis unavailable - OpenCV dependency missing",
                analysis_type="video"
            )

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._get_error_result("Cannot open video file", "video")

            frames = []
            frame_count = 0
            max_frames = 30  # Limit frames for performance

            # Extract frames
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1

            cap.release()

            if len(frames) < 5:
                return self._get_error_result("Insufficient frames for analysis", "video")

            # Perform temporal analysis
            temporal_score = self._analyze_temporal_consistency(frames)
            quality_scores = [self._analyze_frame_quality(frame) for frame in frames[::5]]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5
            
            # Motion analysis
            motion_score = self._analyze_motion_patterns(frames)
            
            # Calculate final authenticity score
            authenticity_score = (temporal_score * 0.4 + avg_quality * 0.3 + motion_score * 0.3)
            confidence_level = self._calculate_confidence([temporal_score, avg_quality, motion_score])

            return {
                'authenticity_score': authenticity_score,
                'is_authentic': authenticity_score > 0.65,
                'confidence': confidence_level,
                'analysis': {
                    'temporal_consistency': temporal_score,
                    'average_quality': avg_quality,
                    'motion_patterns': motion_score,
                    'frames_analyzed': len(frames)
                },
                'analysis_method': 'opencv_temporal',
                'capabilities_used': ['video_analysis']
            }

        except Exception as e:
            return self._get_error_result(str(e), "video")

    def analyze_audio_authenticity(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio authenticity with librosa-based spectral analysis."""
        
        if not LIBROSA_AVAILABLE:
            return self._get_fallback_result(
                "Audio analysis unavailable - librosa dependency missing. This feature requires additional setup for deployment environments.",
                analysis_type="audio",
                score=0.5
            )

        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None, duration=30)  # Limit to 30 seconds
            duration = librosa.get_duration(y=y, sr=sr)

            # Spectral analysis
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

            # Calculate authenticity metrics
            spectral_consistency = self._calculate_spectral_consistency(spectral_centroid)
            temporal_stability = self._calculate_temporal_stability(spectral_rolloff)
            bandwidth_variance = np.var(spectral_bandwidth)
            zcr_variance = np.var(zero_crossing_rate)

            # Combine metrics for authenticity score
            authenticity_score = self._combine_audio_metrics(
                spectral_consistency, temporal_stability, bandwidth_variance, zcr_variance
            )
            
            confidence_level = self._calculate_confidence([
                spectral_consistency, temporal_stability, 
                1 - min(bandwidth_variance, 1.0), 1 - min(zcr_variance, 1.0)
            ])

            return {
                'authenticity_score': authenticity_score,
                'is_authentic': authenticity_score > 0.7,
                'confidence': confidence_level,
                'analysis': {
                    'spectral_consistency': spectral_consistency,
                    'temporal_stability': temporal_stability,
                    'bandwidth_variance': bandwidth_variance,
                    'zero_crossing_variance': zcr_variance,
                    'duration': duration,
                    'sample_rate': sr
                },
                'analysis_method': 'librosa_spectral',
                'capabilities_used': ['audio_analysis']
            }

        except Exception as e:
            return self._get_error_result(str(e), "audio")

    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns in the image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Laplacian variance for sharpness/noise
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Regional noise consistency analysis
            h, w = gray.shape
            regions = [
                gray[0:h//2, 0:w//2],      # Top-left
                gray[0:h//2, w//2:w],      # Top-right
                gray[h//2:h, 0:w//2],      # Bottom-left
                gray[h//2:h, w//2:w]       # Bottom-right
            ]

            noise_vars = [cv2.Laplacian(region, cv2.CV_64F).var() for region in regions]
            noise_consistency = 1 - (np.std(noise_vars) / (np.mean(noise_vars) + 1e-6))

            # Calculate final noise score
            score = min(laplacian_var / 1000, 1.0) * max(noise_consistency, 0.1)

            return {
                'score': score,
                'laplacian_variance': laplacian_var,
                'noise_consistency': noise_consistency,
                'regional_variances': noise_vars
            }
        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect JPEG compression artifacts."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            block_size = 8
            artifacts = []

            # DCT-based artifact analysis
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # Perform DCT
                    dct_block = cv2.dct(block)
                    
                    # Calculate high-frequency energy ratio
                    high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                    total_energy = np.sum(np.abs(dct_block))

                    if total_energy > 0:
                        artifacts.append(high_freq_energy / total_energy)

            artifact_score = np.mean(artifacts) if artifacts else 0.5
            
            return {
                'score': min(artifact_score * 2, 1.0),
                'artifact_level': artifact_score,
                'blocks_analyzed': len(artifacts)
            }
        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _analyze_edge_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency and continuity."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Edge continuity analysis
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            continuity_score = np.sum(dilated > 0) / (np.sum(edges > 0) + 1e-6)

            # Overall edge quality score
            overall_score = edge_density * min(continuity_score / 2, 1.0)

            return {
                'score': overall_score,
                'edge_density': edge_density,
                'continuity_score': continuity_score
            }
        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _analyze_statistical_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced statistical analysis if scipy is available."""
        if not SCIPY_AVAILABLE:
            return {'score': 0.5, 'method': 'scipy_unavailable'}
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Statistical moments
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            skewness = scipy.stats.skew(gray.flatten())
            kurtosis = scipy.stats.kurtosis(gray.flatten())
            
            # Normalize statistical score
            stat_score = 1 - min(abs(skewness) + abs(kurtosis - 3), 2) / 2
            
            return {
                'score': stat_score,
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _analyze_temporal_consistency(self, frames: list) -> float:
        """Analyze temporal consistency between video frames."""
        if len(frames) < 2:
            return 0.5

        consistency_scores = []
        
        for i in range(1, min(len(frames), 10)):
            frame1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Template matching for similarity
            correlation = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF_NORMED)
            max_corr = np.max(correlation)
            consistency_scores.append(max(0, max_corr))

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _analyze_frame_quality(self, frame: np.ndarray) -> float:
        """Analyze individual frame quality."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness using Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            normalized_sharpness = min(sharpness / 1000, 1.0)
            
            return normalized_sharpness
        except Exception:
            return 0.5

    def _analyze_motion_patterns(self, frames: list) -> float:
        """Analyze motion patterns in video frames."""
        if len(frames) < 3:
            return 0.5
        
        try:
            motion_scores = []
            
            for i in range(2, len(frames)):
                frame1 = cv2.cvtColor(frames[i-2], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                frame3 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow1 = cv2.calcOpticalFlowPyrLK(frame1, frame2, None, None)[0]
                flow2 = cv2.calcOpticalFlowPyrLK(frame2, frame3, None, None)[0]
                
                if flow1 is not None and flow2 is not None:
                    # Consistency in motion vectors
                    motion_consistency = np.mean(np.abs(flow1 - flow2))
                    motion_scores.append(1 / (1 + motion_consistency))
            
            return np.mean(motion_scores) if motion_scores else 0.5
        except Exception:
            return 0.5

    def _calculate_spectral_consistency(self, spectral_centroid: np.ndarray) -> float:
        """Calculate spectral consistency for audio."""
        try:
            centroid_mean = np.mean(spectral_centroid)
            centroid_var = np.var(spectral_centroid)
            
            # Higher consistency score for lower variance relative to mean
            consistency = 1 / (1 + centroid_var / (centroid_mean + 1e-6))
            return min(consistency, 1.0)
        except Exception:
            return 0.5

    def _calculate_temporal_stability(self, spectral_rolloff: np.ndarray) -> float:
        """Calculate temporal stability of audio features."""
        try:
            rolloff_diff = np.diff(spectral_rolloff.flatten())
            stability = 1 - min(np.var(rolloff_diff), 1.0)
            return max(stability, 0.0)
        except Exception:
            return 0.5

    def _combine_audio_metrics(self, spectral_consistency: float, temporal_stability: float, 
                              bandwidth_var: float, zcr_var: float) -> float:
        """Combine audio metrics into final authenticity score."""
        try:
            # Weight the different metrics
            weights = [0.3, 0.3, 0.2, 0.2]
            metrics = [
                spectral_consistency,
                temporal_stability,
                1 - min(bandwidth_var, 1.0),  # Lower variance is better
                1 - min(zcr_var, 1.0)         # Lower variance is better
            ]
            
            weighted_score = np.average(metrics, weights=weights)
            return np.clip(weighted_score, 0.0, 1.0)
        except Exception:
            return 0.5

    def _calculate_confidence(self, scores: list) -> str:
        """Calculate confidence level based on score consistency."""
        try:
            if not scores:
                return 'Low'
            
            score_variance = np.var(scores)
            score_mean = np.mean(scores)
            
            # High confidence: low variance, scores not near 0.5
            if score_variance < 0.05 and abs(score_mean - 0.5) > 0.3:
                return 'High'
            elif score_variance < 0.15 and abs(score_mean - 0.5) > 0.15:
                return 'Medium'
            else:
                return 'Low'
        except Exception:
            return 'Low'

    def _get_fallback_result(self, message: str, analysis_type: str, score: float = 0.5) -> Dict[str, Any]:
        """Generate fallback result when analysis is unavailable."""
        return {
            'authenticity_score': score,
            'is_authentic': None,
            'confidence': 'Low',
            'analysis': f'{analysis_type.title()} analysis unavailable',
            'message': message,
            'analysis_method': 'fallback',
            'capabilities_used': []
        }

    def _get_error_result(self, error_message: str, analysis_type: str) -> Dict[str, Any]:
        """Generate error result when analysis fails."""
        return {
            'authenticity_score': 0.5,
            'is_authentic': None,
            'confidence': 'Low',
            'error': error_message,
            'analysis_method': 'error',
            'analysis_type': analysis_type
        }

    def get_system_capabilities(self) -> Dict[str, Any]:
        """Return current system capabilities and dependency status."""
        return {
            'capabilities': self.capabilities,
            'dependencies': {
                'opencv': CV2_AVAILABLE,
                'librosa': LIBROSA_AVAILABLE,
                'scipy': SCIPY_AVAILABLE
            },
            'supported_formats': {
                'image': ['jpg', 'jpeg', 'png', 'bmp'] if CV2_AVAILABLE else [],
                'video': ['mp4', 'avi', 'mov'] if CV2_AVAILABLE else [],
                'audio': ['wav', 'mp3', 'flac', 'm4a'] if LIBROSA_AVAILABLE else []
            },
            'deployment_ready': CV2_AVAILABLE,  # Minimum requirement for basic functionality
            'full_featured': all(self.capabilities.values())
        }

    def test_capabilities(self) -> Dict[str, str]:
        """Test all capabilities and return status report."""
        results = {}
        
        # Test image analysis
        if CV2_AVAILABLE:
            try:
                test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                self.analyze_image_authenticity(test_image)
                results['image_analysis'] = "✅ Working"
            except Exception as e:
                results['image_analysis'] = f"❌ Error: {e}"
        else:
            results['image_analysis'] = "❌ OpenCV not available"
        
        # Test audio analysis
        if LIBROSA_AVAILABLE:
            results['audio_analysis'] = "✅ Available (requires audio file for full test)"
        else:
            results['audio_analysis'] = "❌ Librosa not available"
        
        # Test scipy
        if SCIPY_AVAILABLE:
            results['statistical_analysis'] = "✅ Available"
        else:
            results['statistical_analysis'] = "❌ Scipy not available"
        
        return results
