"""
Test script for the enhanced spatial clustering module
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the spatial_analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_analysis.enhanced_spatial_clustering import EnhancedSpatialClustering, SpatialClusteringParameters

def test_enhanced_spatial_clustering():
    """Test the enhanced spatial clustering module"""
    print("Testing Enhanced Spatial Clustering...")
    
    try:
        # Load actual data using our enhanced data processor
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_processing.enhanced_data_processor import EnhancedDataProcessor
        
        # Initialize the data processor
        processor = EnhancedDataProcessor()
        
        # Load and clean state data
        state_data = processor.load_and_clean_state_data()
        
        print(f"\n1. Loaded data successfully:")
        print(f"   State data shape: {state_data.shape}")
        
        # Initialize the enhanced spatial clustering model
        model = EnhancedSpatialClustering(state_data)
        
        # Test spatial summary
        print("\n2. Testing spatial summary...")
        summary = model.get_spatial_summary()
        print(f"   Total states: {summary['total_states']}")
        print(f"   Mean illegal incidence: {summary['mean_illegal_incidence']:.1f}%")
        print(f"   Moran's I: {summary['morans_i']:.3f}")
        
        # Test hotspot detection
        print("\n3. Testing hotspot detection...")
        hotspot_results = model.detect_hotspots()
        hotspot_count = hotspot_results['combined_hotspot'].sum()
        print(f"   Hotspots detected: {hotspot_count}")
        
        print("   Hotspot states:")
        hotspots = hotspot_results[hotspot_results['combined_hotspot']]
        for _, row in hotspots.iterrows():
            print(f"     {row['state']}: {row['illegal_incidence']:.1f}%")
        
        # Test clustering methods comparison
        print("\n4. Testing clustering methods comparison...")
        comparison = model.compare_clustering_methods()
        print(f"   Best method: {comparison['best_method']}")
        print(f"   Best silhouette score: {comparison['best_score']:.3f}")
        
        # Test individual clustering method
        print("\n5. Testing individual clustering method...")
        best_method = comparison['best_method']
        clustering_results = model.cluster_spatial_data(method=best_method)
        
        if best_method in clustering_results:
            result = clustering_results[best_method]
            print(f"   {best_method.upper()} clustering completed")
            print(f"   Number of clusters: {result['n_clusters']}")
            if 'silhouette_score' in result:
                print(f"   Silhouette score: {result['silhouette_score']:.3f}")
            
            # Show cluster assignments
            print("   Cluster assignments:")
            clusters = result['clusters']
            for i, cluster in enumerate(clusters):
                state_name = hotspot_results.iloc[i]['state']
                print(f"     {state_name}: Cluster {cluster}")
        
        # Test parameter updates
        print("\n6. Testing parameter updates...")
        original_params = model.get_model_parameters()
        print(f"   Original DBSCAN eps: {original_params['dbscan_eps']}")
        
        # Update a parameter
        model.update_parameters({'dbscan_eps': 1.0})
        updated_params = model.get_model_parameters()
        print(f"   Updated DBSCAN eps: {updated_params['dbscan_eps']}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_spatial_clustering()
