# TDA Platform Integration Guides

This section provides comprehensive integration guides for building end-to-end TDA workflows, integrating with third-party tools, and deploying the platform in production environments.

## 📚 Integration Documentation Structure

### 🔄 [End-to-End Workflows](./workflows/)
- **Complete Analysis Pipelines** - From data ingestion to visualization
- **Batch Processing** - Large-scale dataset processing workflows
- **Real-time Streaming** - Live data analysis with Kafka/Flink
- **Interactive Analysis** - Jupyter notebook workflows

### 🛠️ [Third-Party Tools](./third-party/)
- **GUDHI Integration** - Advanced TDA library integration
- **CGAL Integration** - Geometric kernel configuration
- **NumPy/Pandas** - Data processing and analysis
- **Visualization Tools** - Plotly, D3.js, and matplotlib integration

### 🚀 [Deployment & Production](./deployment/)
- **Docker Containers** - Containerized deployment
- **Kubernetes** - Orchestration and scaling
- **Cloud Platforms** - AWS, GCP, Azure deployment
- **Monitoring & Logging** - Production observability

### 🔗 [Data Integration](./data/)
- **Data Formats** - CSV, JSON, HDF5, Parquet support
- **Database Integration** - PostgreSQL, MongoDB workflows
- **Streaming Sources** - Kafka, Kinesis, Pub/Sub integration
- **ETL Pipelines** - Data transformation and loading

## 🚀 Quick Integration Examples

### 1. **Basic Python Integration**
```python
# Complete TDA analysis workflow
import numpy as np
import pandas as pd
from tda_backend import TDAEngine, PointCloud
import plotly.graph_objects as go

# Load data
df = pd.read_csv("financial_data.csv")
points = PointCloud.from_dataframe(df, columns=["price", "volume", "volatility"])

# Initialize engine
engine = TDAEngine()

# Run analysis
results = engine.compute_persistent_homology(
    points,
    method="vietoris_rips",
    max_dimension=2,
    parameters={"epsilon": 0.1}
)

# Visualize results
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[pair.birth for pair in results.persistence_pairs if pair.dimension == 1],
    y=[pair.death for pair in results.persistence_pairs if pair.dimension == 1],
    mode='markers',
    name='H1 Features'
))
fig.update_layout(title="Persistence Diagram", xaxis_title="Birth", yaxis_title="Death")
fig.show()
```

### 2. **GUDHI Integration**
```python
# Advanced TDA with GUDHI
import gudhi
from tda_backend import TDAEngine
import numpy as np

# Create synthetic data
points = np.random.rand(1000, 3)

# Use TDA Platform for initial computation
engine = TDAEngine()
platform_results = engine.compute_persistent_homology(
    points, method="vietoris_rips", max_dimension=2
)

# Compare with GUDHI
gudhi_complex = gudhi.RipsComplex(points=points, max_edge_length=0.5)
gudhi_simplex_tree = gudhi_complex.create_simplex_tree(max_dimension=2)
gudhi_results = gudhi_simplex_tree.persistence()

# Validate results
print(f"Platform: {len(platform_results.persistence_pairs)} pairs")
print(f"GUDHI: {len(gudhi_results)} pairs")

# Use GUDHI for advanced analysis
wasserstein_distance = gudhi.wasserstein.wasserstein_distance(
    platform_results.persistence_pairs,
    gudhi_results,
    order=2
)
print(f"Wasserstein distance: {wasserstein_distance}")
```

### 3. **Real-time Streaming Integration**
```python
# Kafka streaming integration
from kafka import KafkaConsumer, KafkaProducer
import json
from tda_backend import TDAEngine
import numpy as np

class StreamingTDAAnalyzer:
    def __init__(self):
        self.engine = TDAEngine()
        self.consumer = KafkaConsumer(
            'network-traffic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def process_stream(self):
        window_data = []
        
        for message in self.consumer:
            # Add to sliding window
            window_data.append(message.value['features'])
            
            # Process when window is full
            if len(window_data) >= 1000:
                self.analyze_window(window_data)
                window_data = window_data[100:]  # Sliding window
                
    def analyze_window(self, data):
        points = np.array(data)
        
        # Run TDA analysis
        results = self.engine.compute_persistent_homology(
            points, method="vietoris_rips", max_dimension=1
        )
        
        # Send results to output topic
        self.producer.send('tda-results', {
            'timestamp': time.time(),
            'persistence_pairs': len(results.persistence_pairs),
            'anomaly_score': self.compute_anomaly_score(results)
        })

# Start streaming analysis
analyzer = StreamingTDAAnalyzer()
analyzer.process_stream()
```

## 🔄 End-to-End Workflow Examples

### Financial Market Analysis Pipeline
```python
# Complete financial analysis workflow
class FinancialAnalysisPipeline:
    def __init__(self):
        self.engine = TDAEngine()
        self.db = DatabaseConnection()
        
    def run_analysis(self, start_date, end_date, symbols):
        # 1. Data Ingestion
        market_data = self.fetch_market_data(start_date, end_date, symbols)
        
        # 2. Preprocessing
        point_clouds = self.create_point_clouds(market_data)
        
        # 3. TDA Analysis
        results = {}
        for symbol, points in point_clouds.items():
            results[symbol] = self.engine.compute_persistent_homology(
                points, method="vietoris_rips", max_dimension=2
            )
        
        # 4. Feature Extraction
        features = self.extract_topological_features(results)
        
        # 5. Regime Detection
        regimes = self.detect_market_regimes(features)
        
        # 6. Storage and Visualization
        self.store_results(results, features, regimes)
        self.generate_reports(regimes)
        
        return regimes
    
    def create_point_clouds(self, market_data):
        """Convert time series to point clouds using sliding windows"""
        point_clouds = {}
        
        for symbol, data in market_data.items():
            # Create sliding window embeddings
            windows = self.create_sliding_windows(data, window_size=20, step_size=5)
            
            # Convert to point clouds
            point_clouds[symbol] = np.array(windows)
            
        return point_clouds
    
    def extract_topological_features(self, results):
        """Extract vectorized features from persistence results"""
        features = {}
        
        for symbol, result in results.items():
            # Compute persistence landscapes
            landscapes = result.compute_persistence_landscapes(
                num_landscapes=10, resolution=100
            )
            
            # Compute persistence images
            images = result.compute_persistence_images(
                resolution=50, bandwidth=0.1
            )
            
            # Compute Betti curves
            betti_curves = result.compute_betti_curves(
                num_points=100, max_epsilon=2.0
            )
            
            features[symbol] = {
                'landscapes': landscapes,
                'images': images,
                'betti_curves': betti_curves
            }
            
        return features
```

### Cybersecurity Anomaly Detection Pipeline
```python
# Network traffic anomaly detection
class NetworkAnomalyDetector:
    def __init__(self):
        self.engine = TDAEngine()
        self.baseline_model = self.load_baseline_model()
        
    def detect_anomalies(self, network_traffic):
        # 1. Packet preprocessing
        features = self.extract_packet_features(network_traffic)
        
        # 2. Create time-based windows
        windows = self.create_time_windows(features, window_size=60)  # 60 seconds
        
        # 3. Compute topological features for each window
        topological_features = []
        for window in windows:
            # Convert to point cloud
            points = self.convert_to_point_cloud(window)
            
            # Run TDA analysis
            results = self.engine.compute_persistent_homology(
                points, method="vietoris_rips", max_dimension=1
            )
            
            # Extract features
            window_features = self.extract_window_features(results)
            topological_features.append(window_features)
        
        # 4. Anomaly detection
        anomalies = self.detect_anomalies_from_features(topological_features)
        
        # 5. Alert generation
        if anomalies:
            self.generate_alerts(anomalies)
            
        return anomalies
    
    def extract_packet_features(self, traffic):
        """Extract relevant features from network packets"""
        features = []
        
        for packet in traffic:
            feature_vector = [
                packet.length,
                packet.duration,
                packet.protocol,
                packet.src_port,
                packet.dst_port,
                packet.packet_count,
                packet.byte_count
            ]
            features.append(feature_vector)
            
        return np.array(features)
```

## 🛠️ Third-Party Tool Integration

### GUDHI Library Integration
```python
# Advanced GUDHI integration
import gudhi
from tda_backend import TDAEngine
import numpy as np

class GUDHIIntegration:
    def __init__(self):
        self.engine = TDAEngine()
        
    def compare_algorithms(self, points):
        """Compare TDA Platform with GUDHI implementations"""
        
        # TDA Platform
        platform_start = time.time()
        platform_results = self.engine.compute_persistent_homology(
            points, method="vietoris_rips", max_dimension=2
        )
        platform_time = time.time() - platform_start
        
        # GUDHI
        gudhi_start = time.time()
        gudhi_complex = gudhi.RipsComplex(points=points, max_edge_length=0.5)
        gudhi_simplex_tree = gudhi_complex.create_simplex_tree(max_dimension=2)
        gudhi_results = gudhi_simplex_tree.persistence()
        gudhi_time = time.time() - gudhi_start
        
        # Performance comparison
        comparison = {
            'platform_time': platform_time,
            'gudhi_time': gudhi_time,
            'platform_pairs': len(platform_results.persistence_pairs),
            'gudhi_pairs': len(gudhi_results),
            'speedup': gudhi_time / platform_time
        }
        
        return comparison
    
    def hybrid_analysis(self, points):
        """Use both platforms for comprehensive analysis"""
        
        # Quick analysis with TDA Platform
        quick_results = self.engine.compute_persistent_homology(
            points, method="vietoris_rips", max_dimension=1
        )
        
        # Detailed analysis with GUDHI
        gudhi_complex = gudhi.RipsComplex(points=points, max_edge_length=0.5)
        gudhi_simplex_tree = gudhi_complex.create_simplex_tree(max_dimension=2)
        
        # Advanced GUDHI features
        gudhi_persistence = gudhi_simplex_tree.persistence()
        gudhi_betti_numbers = gudhi_simplex_tree.betti_numbers()
        gudhi_euler_characteristic = gudhi_simplex_tree.euler_characteristic()
        
        return {
            'quick_analysis': quick_results,
            'detailed_analysis': {
                'persistence': gudhi_persistence,
                'betti_numbers': gudhi_betti_numbers,
                'euler_characteristic': gudhi_euler_characteristic
            }
        }
```

### CGAL Geometric Kernel Integration
```cpp
// CGAL integration for geometric operations
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <tda/algorithms/alpha_complex.hpp>

class CGALIntegration {
private:
    typedef CGAL::Simple_cartesian<double> Kernel;
    typedef Kernel::Point_2 Point_2;
    typedef CGAL::Alpha_shape_vertex_base_2<Kernel> Vb;
    typedef CGAL::Alpha_shape_face_base_2<Kernel> Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
    typedef CGAL::Alpha_shape_2<Kernel, Tds> Alpha_shape_2;
    
public:
    std::vector<Simplex> compute_alpha_complex_cgal(
        const std::vector<Point>& points,
        double alpha_value
    ) {
        std::vector<Point_2> cgal_points;
        cgal_points.reserve(points.size());
        
        // Convert to CGAL points
        for (const auto& p : points) {
            cgal_points.emplace_back(p.x, p.y);
        }
        
        // Compute alpha shape
        Alpha_shape_2 alpha_shape(cgal_points.begin(), cgal_points.end(),
                                 alpha_value, Alpha_shape_2::GENERAL);
        
        // Extract simplices
        std::vector<Simplex> simplices;
        
        // Vertices (0-simplices)
        for (auto vit = alpha_shape.vertices_begin(); 
             vit != alpha_shape.vertices_end(); ++vit) {
            if (alpha_shape.classify(vit) == Alpha_shape_2::REGULAR ||
                alpha_shape.classify(vit) == Alpha_shape_2::SINGULAR) {
                simplices.emplace_back(0, {vit->point().x(), vit->point().y()});
            }
        }
        
        // Edges (1-simplices)
        for (auto eit = alpha_shape.edges_begin(); 
             eit != alpha_shape.edges_end(); ++eit) {
            if (alpha_shape.classify(eit) == Alpha_shape_2::REGULAR ||
                alpha_shape.classify(eit) == Alpha_shape_2::SINGULAR) {
                // Extract edge vertices
                auto v1 = eit->first->vertex(alpha_shape.ccw(eit->second));
                auto v2 = eit->first->vertex(alpha_shape.cw(eit->second));
                
                simplices.emplace_back(1, {
                    {v1->point().x(), v1->point().y()},
                    {v2->point().x(), v2->point().y()}
                });
            }
        }
        
        return simplices;
    }
};
```

## 🚀 Deployment Integration

### Docker Compose Setup
```yaml
# docker-compose.yml for complete TDA platform
version: '3.8'

services:
  tda-backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tda
      - MONGODB_URL=mongodb://mongo:27017/tda
    depends_on:
      - postgres
      - mongo
      - kafka
      - redis

  tda-frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - tda-backend

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=tda
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mongo:
    image: mongo:5
    volumes:
      - mongo_data:/data/db

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

volumes:
  postgres_data:
  mongo_data:
  grafana_data:
  prometheus_data:
```

### Kubernetes Deployment
```yaml
# kubernetes/tda-platform.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tda-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tda-backend
  template:
    metadata:
      labels:
        app: tda-backend
    spec:
      containers:
      - name: tda-backend
        image: tda-platform/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tda-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: tda-backend-service
spec:
  selector:
    app: tda-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tda-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tda-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔗 Related Documentation

- **[API Reference](../api/)** - Integration-focused API usage
- **[Performance Guide](../performance/)** - Optimization for integration
- **[Troubleshooting](../troubleshooting/)** - Integration issues
- **[Examples](../examples/)** - Complete integration examples

## 📞 Integration Support

- **Integration Issues**: Include error logs and environment details
- **Workflow Questions**: Describe your use case and requirements
- **Deployment Help**: Specify target environment and constraints

---

*Ready to integrate? Start with [End-to-End Workflows](./workflows/) for complete pipelines, or check [Third-Party Tools](./third-party/) for specific integrations.*
