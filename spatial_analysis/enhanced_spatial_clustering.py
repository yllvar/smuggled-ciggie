"""
Enhanced Spatial Clustering Module for Malaysia Illicit Cigarettes Study

This module provides improved spatial clustering algorithms and hotspot detection
compared to the original implementation.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpatialClusteringParameters:
    """Data class for spatial clustering parameters"""
    # DBSCAN parameters
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 2
    
    # K-means parameters
    kmeans_n_clusters: int = 3
    
    # Agglomerative clustering parameters
    agg_n_clusters: int = 3
    agg_linkage: str = 'ward'
    
    # Hotspot detection parameters
    hotspot_threshold_percentile: float = 75.0
    spatial_weight_method: str = 'inverse_distance'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
            'kmeans_n_clusters': self.kmeans_n_clusters,
            'agg_n_clusters': self.agg_n_clusters,
            'agg_linkage': self.agg_linkage,
            'hotspot_threshold_percentile': self.hotspot_threshold_percentile,
            'spatial_weight_method': self.spatial_weight_method
        }

class EnhancedSpatialClustering:
    """Enhanced spatial clustering with multiple algorithms and improved hotspot detection"""
    
    def __init__(self, state_data: pd.DataFrame, 
                 parameters: Optional[SpatialClusteringParameters] = None):
        """
        Initialize the enhanced spatial clustering model
        
        Args:
            state_data: DataFrame with state-level data including illegal incidence
            parameters: Clustering parameters (uses defaults if None)
        """
        self.state_data = state_data.copy()
        self.parameters = parameters or SpatialClusteringParameters()
        
        # Validate input data
        self._validate_data()
        
        # Create spatial coordinates
        self._create_spatial_coordinates()
        
        # Calculate derived spatial data
        self._calculate_spatial_data()
        
        logger.info("Enhanced Spatial Clustering initialized successfully")
    
    def _validate_data(self):
        """Validate input data"""
        required_columns = [
            'State', 'Incidence of illegal cigarettes'
        ]
        
        for col in required_columns:
            if col not in self.state_data.columns:
                raise ValueError(f"Required column '{col}' not found in state data")
        
        if not self.state_data['State'].is_unique:
            logger.warning("Duplicate state names found in data")
        
        # Validate incidence values are reasonable
        illegal_incidence = self.state_data['Incidence of illegal cigarettes']
        if not ((illegal_incidence >= 0) & (illegal_incidence <= 100)).all():
            raise ValueError("Illegal incidence values must be between 0 and 100")
        
        logger.info("Data validation completed successfully")
    
    def _create_spatial_coordinates(self):
        """Create spatial coordinates for Malaysian states (approximate centroids)"""
        # More accurate coordinates for Malaysian states
        self.state_coordinates = {
            'PERLIS': (6.4414, 100.1986),
            'KEDAH': (6.1184, 100.3685),
            'PENANG': (5.4164, 100.3327),
            'PERAK': (4.5921, 101.0901),
            'SELANGOR': (3.0738, 101.5183),
            'WP KL': (3.1390, 101.6869),
            'N.SEMBILAN': (2.7258, 101.9424),
            'MELAKA': (2.1896, 102.2501),
            'JOHOR': (1.4927, 103.7414),
            'PAHANG': (3.8126, 103.3256),
            'T\'GANU': (5.3117, 103.1324),
            'KELANTAN': (6.1254, 102.2386),
            'SABAH': (5.9804, 116.0735),
            'SARAWAK': (1.5533, 110.3592)
        }
        
        logger.debug("Spatial coordinates created for Malaysian states")
    
    def _calculate_spatial_data(self):
        """Calculate derived spatial data"""
        # Create spatial dataset
        spatial_data = []
        for _, row in self.state_data.iterrows():
            state_name = row['State']
            if state_name in self.state_coordinates:
                lat, lon = self.state_coordinates[state_name]
                spatial_data.append({
                    'state': state_name,
                    'latitude': lat,
                    'longitude': lon,
                    'illegal_incidence': row['Incidence of illegal cigarettes'],
                })
        
        self.spatial_df = pd.DataFrame(spatial_data)
        
        if self.spatial_df.empty:
            raise ValueError("No valid spatial data could be created")
        
        # Calculate distance matrix
        coords = self.spatial_df[['latitude', 'longitude']].values
        self.distance_matrix = squareform(pdist(coords, metric='euclidean'))
        
        # Create spatial weights matrix
        self.spatial_weights = self._create_spatial_weights()
        
        logger.debug(f"Spatial data calculated: {len(spatial_data)} states")
    
    def _create_spatial_weights(self) -> np.ndarray:
        """
        Create spatial weights matrix using specified method
        
        Returns:
            Spatial weights matrix
        """
        weights = np.zeros_like(self.distance_matrix)
        
        if self.parameters.spatial_weight_method == 'inverse_distance':
            # Inverse distance weighting
            for i in range(len(self.distance_matrix)):
                for j in range(len(self.distance_matrix)):
                    if i != j and self.distance_matrix[i, j] > 0:
                        weights[i, j] = 1 / self.distance_matrix[i, j]
        elif self.parameters.spatial_weight_method == 'binary':
            # Binary weights (1 if within threshold distance, 0 otherwise)
            threshold = np.percentile(self.distance_matrix[self.distance_matrix > 0], 50)
            for i in range(len(self.distance_matrix)):
                for j in range(len(self.distance_matrix)):
                    if i != j and self.distance_matrix[i, j] <= threshold:
                        weights[i, j] = 1
        
        # Normalize weights
        row_sums = weights.sum(axis=1)
        for i in range(len(weights)):
            if row_sums[i] > 0:
                weights[i] = weights[i] / row_sums[i]
        
        return weights
    
    def calculate_morans_i(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate Global Moran's I statistic for spatial autocorrelation
        
        Args:
            values: Array of values to analyze
            weights: Spatial weights matrix (uses self.spatial_weights if None)
            
        Returns:
            Moran's I statistic
        """
        if weights is None:
            weights = self.spatial_weights
        
        n = len(values)
        mean_val = np.mean(values)
        
        # Numerator: sum of spatial autocovariance
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        
        # Denominator: variance * sum of weights
        variance = np.sum((values - mean_val) ** 2) / n
        weight_sum = np.sum(weights)
        
        if weight_sum > 0 and variance > 0:
            morans_i = numerator / (weight_sum * variance)
        else:
            morans_i = 0
        
        return morans_i
    
    def detect_hotspots(self) -> pd.DataFrame:
        """
        Detect hotspots using multiple methods
        
        Returns:
            DataFrame with hotspot detection results
        """
        try:
            hotspot_results = []
            
            # Get illegal incidence values
            incidence_values = self.spatial_df['illegal_incidence'].values
            
            # 1. Percentile-based hotspot detection
            threshold = np.percentile(incidence_values, self.parameters.hotspot_threshold_percentile)
            percentile_hotspots = incidence_values >= threshold
            
            # 2. Z-score-based hotspot detection
            z_scores = zscore(incidence_values)
            z_score_hotspots = z_scores >= 1.0  # Above 1 standard deviation
            
            # 3. Moran's I-based hotspot detection
            morans_i = self.calculate_morans_i(incidence_values)
            # If positive spatial autocorrelation, enhance hotspot detection
            spatial_adjustment = 1.0 + max(0, morans_i)  # Increase threshold if clustering detected
            adjusted_threshold = threshold * spatial_adjustment
            morans_hotspots = incidence_values >= adjusted_threshold
            
            # Combine methods
            combined_hotspots = (percentile_hotspots & z_score_hotspots) | morans_hotspots
            
            # Add results to dataframe
            self.spatial_df['percentile_hotspot'] = percentile_hotspots
            self.spatial_df['z_score_hotspot'] = z_score_hotspots
            self.spatial_df['morans_hotspot'] = morans_hotspots
            self.spatial_df['combined_hotspot'] = combined_hotspots
            self.spatial_df['z_score'] = z_scores
            self.spatial_df['hotspot_threshold'] = threshold
            self.spatial_df['morans_i'] = morans_i
            
            logger.info(f"Hotspot detection completed: {combined_hotspots.sum()} hotspots identified")
            return self.spatial_df
            
        except Exception as e:
            logger.error(f"Error in hotspot detection: {str(e)}")
            raise
    
    def cluster_spatial_data(self, method: str = 'all') -> Dict[str, Any]:
        """
        Apply multiple clustering algorithms to spatial data
        
        Args:
            method: Clustering method ('dbscan', 'kmeans', 'agglomerative', 'all')
            
        Returns:
            Dictionary with clustering results
        """
        try:
            results = {}
            
            # Prepare features for clustering
            features = self.spatial_df[['latitude', 'longitude', 'illegal_incidence']].copy()
            features['illegal_incidence_scaled'] = features['illegal_incidence'] / 100  # Scale to 0-1
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features[['latitude', 'longitude', 'illegal_incidence_scaled']])
            
            # Apply selected clustering methods
            if method in ['dbscan', 'all']:
                logger.info("Applying DBSCAN clustering...")
                dbscan = DBSCAN(eps=self.parameters.dbscan_eps, 
                               min_samples=self.parameters.dbscan_min_samples)
                dbscan_clusters = dbscan.fit_predict(features_scaled)
                
                # Calculate silhouette score if more than 1 cluster
                n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
                if n_clusters > 1:
                    # Only calculate silhouette for non-noise points
                    non_noise_mask = dbscan_clusters != -1
                    if non_noise_mask.sum() > 1:
                        dbscan_silhouette = silhouette_score(
                            features_scaled[non_noise_mask], 
                            dbscan_clusters[non_noise_mask]
                        )
                    else:
                        dbscan_silhouette = 0
                else:
                    dbscan_silhouette = 0
                
                results['dbscan'] = {
                    'clusters': dbscan_clusters,
                    'n_clusters': n_clusters,
                    'n_noise_points': list(dbscan_clusters).count(-1),
                    'silhouette_score': dbscan_silhouette
                }
                
                logger.debug(f"DBSCAN: {n_clusters} clusters, {list(dbscan_clusters).count(-1)} noise points")
            
            if method in ['kmeans', 'all']:
                logger.info("Applying K-means clustering...")
                kmeans = KMeans(n_clusters=self.parameters.kmeans_n_clusters, random_state=42)
                kmeans_clusters = kmeans.fit_predict(features_scaled)
                
                # Calculate silhouette score
                kmeans_silhouette = silhouette_score(features_scaled, kmeans_clusters)
                
                results['kmeans'] = {
                    'clusters': kmeans_clusters,
                    'n_clusters': self.parameters.kmeans_n_clusters,
                    'silhouette_score': kmeans_silhouette,
                    'inertia': kmeans.inertia_
                }
                
                logger.debug(f"K-means: {self.parameters.kmeans_n_clusters} clusters, silhouette: {kmeans_silhouette:.3f}")
            
            if method in ['agglomerative', 'all']:
                logger.info("Applying Agglomerative clustering...")
                agg = AgglomerativeClustering(
                    n_clusters=self.parameters.agg_n_clusters, 
                    linkage=self.parameters.agg_linkage
                )
                agg_clusters = agg.fit_predict(features_scaled)
                
                # Calculate silhouette score
                agg_silhouette = silhouette_score(features_scaled, agg_clusters)
                
                results['agglomerative'] = {
                    'clusters': agg_clusters,
                    'n_clusters': self.parameters.agg_n_clusters,
                    'silhouette_score': agg_silhouette
                }
                
                logger.debug(f"Agglomerative: {self.parameters.agg_n_clusters} clusters, silhouette: {agg_silhouette:.3f}")
            
            logger.info(f"Spatial clustering completed using method: {method}")
            return results
            
        except Exception as e:
            logger.error(f"Error in spatial clustering: {str(e)}")
            raise
    
    def compare_clustering_methods(self) -> Dict[str, Any]:
        """
        Compare different clustering methods and recommend the best one
        
        Returns:
            Dictionary with comparison results and recommendation
        """
        try:
            # Apply all clustering methods
            clustering_results = self.cluster_spatial_data(method='all')
            
            # Compare silhouette scores
            method_scores = {}
            for method, result in clustering_results.items():
                method_scores[method] = result.get('silhouette_score', 0)
            
            # Recommend best method based on silhouette score
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            
            comparison_results = {
                'method_scores': method_scores,
                'best_method': best_method,
                'best_score': best_score,
                'recommendation': f"Use {best_method} clustering (silhouette score: {best_score:.3f})"
            }
            
            logger.info(f"Clustering comparison completed. Recommended method: {best_method}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in clustering comparison: {str(e)}")
            raise
    
    def get_spatial_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for spatial analysis
        
        Returns:
            Dictionary with spatial summary statistics
        """
        try:
            incidence_values = self.spatial_df['illegal_incidence'].values
            
            summary = {
                'total_states': len(self.spatial_df),
                'mean_illegal_incidence': np.mean(incidence_values),
                'std_illegal_incidence': np.std(incidence_values),
                'min_illegal_incidence': np.min(incidence_values),
                'max_illegal_incidence': np.max(incidence_values),
                'morans_i': self.calculate_morans_i(incidence_values),
                'hotspot_count': self.spatial_df['combined_hotspot'].sum() if 'combined_hotspot' in self.spatial_df.columns else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in spatial summary: {str(e)}")
            raise
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters
        
        Returns:
            Dictionary with current model parameters
        """
        return self.parameters.to_dict()
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """
        Update model parameters
        
        Args:
            new_parameters: Dictionary with new parameter values
        """
        try:
            # Update parameters
            for key, value in new_parameters.items():
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, value)
                    logger.debug(f"Updated parameter '{key}' to {value}")
                else:
                    logger.warning(f"Unknown parameter '{key}' ignored")
            
            # Recalculate spatial weights if needed
            if 'spatial_weight_method' in new_parameters:
                self.spatial_weights = self._create_spatial_weights()
            
            logger.info("Model parameters updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise

def main():
    """Main function to demonstrate the enhanced spatial clustering"""
    try:
        # This would normally load actual data
        # For demonstration, we'll create sample data
        
        # Sample state data
        state_data = pd.DataFrame({
            'State': ['PERLIS', 'KEDAH', 'PENANG', 'PERAK'],
            'Incidence of illegal cigarettes': [41.8, 32.4, 47.8, 33.2]
        })
        
        # Initialize the model
        model = EnhancedSpatialClustering(state_data)
        
        # Get spatial summary
        summary = model.get_spatial_summary()
        print("Spatial Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Detect hotspots
        hotspot_results = model.detect_hotspots()
        print("\nHotspot Detection Results:")
        print(hotspot_results[['state', 'illegal_incidence', 'combined_hotspot']].to_string(index=False))
        
        # Compare clustering methods
        comparison = model.compare_clustering_methods()
        print("\nClustering Method Comparison:")
        for method, score in comparison['method_scores'].items():
            print(f"  {method}: {score:.3f}")
        print(f"\nRecommendation: {comparison['recommendation']}")
        
        # Apply recommended clustering method
        best_method = comparison['best_method']
        clustering_results = model.cluster_spatial_data(method=best_method)
        
        print(f"\n{best_method.title()} Clustering Results:")
        clusters = clustering_results[best_method]['clusters']
        for i, cluster in enumerate(clusters):
            print(f"  {hotspot_results.iloc[i]['state']}: Cluster {cluster}")
        
        print("\n✅ Enhanced Spatial Clustering demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
