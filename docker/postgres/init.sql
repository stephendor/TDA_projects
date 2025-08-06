-- TDA Platform Database Initialization Script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS tda_core;
CREATE SCHEMA IF NOT EXISTS cybersecurity;
CREATE SCHEMA IF NOT EXISTS finance;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Core TDA computation results
CREATE TABLE IF NOT EXISTS tda_core.computation_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    input_data_hash VARCHAR(64) NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result_data JSONB,
    computation_time_ms INTEGER
);

CREATE INDEX idx_computation_jobs_status ON tda_core.computation_jobs(status);
CREATE INDEX idx_computation_jobs_created_at ON tda_core.computation_jobs(created_at);
CREATE INDEX idx_computation_jobs_job_type ON tda_core.computation_jobs(job_type);

-- Persistent homology results cache
CREATE TABLE IF NOT EXISTS tda_core.persistence_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_hash VARCHAR(64) UNIQUE NOT NULL,
    maxdim INTEGER NOT NULL,
    persistence_diagram JSONB NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_persistence_cache_hash ON tda_core.persistence_cache(data_hash);
CREATE INDEX idx_persistence_cache_accessed ON tda_core.persistence_cache(last_accessed);

-- Cybersecurity threat detection results
CREATE TABLE IF NOT EXISTS cybersecurity.threat_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_ip INET,
    source_system VARCHAR(100),
    threat_level VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    threat_indicators JSONB NOT NULL,
    topological_features JSONB,
    mitigation_status VARCHAR(20) DEFAULT 'pending',
    investigated_at TIMESTAMP WITH TIME ZONE,
    investigator_id VARCHAR(50)
);

CREATE INDEX idx_threat_detections_timestamp ON cybersecurity.threat_detections(timestamp);
CREATE INDEX idx_threat_detections_type ON cybersecurity.threat_detections(detection_type);
CREATE INDEX idx_threat_detections_level ON cybersecurity.threat_detections(threat_level);

-- APT detection results
CREATE TABLE IF NOT EXISTS cybersecurity.apt_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    time_window_hours INTEGER NOT NULL,
    network_segment VARCHAR(100),
    apt_percentage DECIMAL(5,2) NOT NULL,
    high_risk_samples INTEGER NOT NULL,
    threat_assessment VARCHAR(20) NOT NULL,
    persistence_features JSONB NOT NULL,
    temporal_patterns JSONB,
    recommended_actions TEXT[]
);

-- Financial risk assessments
CREATE TABLE IF NOT EXISTS finance.risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    portfolio_id VARCHAR(100) NOT NULL,
    risk_type VARCHAR(50) NOT NULL,
    overall_risk_score INTEGER NOT NULL,
    var_1day DECIMAL(12,4),
    var_10day DECIMAL(12,4),
    expected_shortfall DECIMAL(12,4),
    topological_risk_features JSONB NOT NULL,
    stress_test_results JSONB,
    correlation_analysis JSONB
);

CREATE INDEX idx_risk_assessments_timestamp ON finance.risk_assessments(assessment_timestamp);
CREATE INDEX idx_risk_assessments_portfolio ON finance.risk_assessments(portfolio_id);

-- Cryptocurrency bubble detection
CREATE TABLE IF NOT EXISTS finance.bubble_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    asset_symbol VARCHAR(20) NOT NULL,
    bubble_probability DECIMAL(4,3) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    recommendation VARCHAR(50) NOT NULL,
    topological_indicators JSONB NOT NULL,
    price_data_window_days INTEGER NOT NULL,
    detection_horizon_days INTEGER NOT NULL
);

-- System monitoring and performance
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    component VARCHAR(50) NOT NULL,
    tags JSONB
);

CREATE INDEX idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp);
CREATE INDEX idx_system_metrics_component ON monitoring.system_metrics(component);

-- API usage tracking
CREATE TABLE IF NOT EXISTS monitoring.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    user_id VARCHAR(100),
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    error_message TEXT
);

CREATE INDEX idx_api_requests_timestamp ON monitoring.api_requests(timestamp);
CREATE INDEX idx_api_requests_endpoint ON monitoring.api_requests(endpoint);

-- Create database users and permissions
DO $$
BEGIN
    -- API application user
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'tda_api_user') THEN
        CREATE ROLE tda_api_user WITH LOGIN PASSWORD 'tda_api_password';
    END IF;
    
    -- Worker process user
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'tda_worker_user') THEN
        CREATE ROLE tda_worker_user WITH LOGIN PASSWORD 'tda_worker_password';
    END IF;
    
    -- Read-only monitoring user
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'tda_monitor_user') THEN
        CREATE ROLE tda_monitor_user WITH LOGIN PASSWORD 'tda_monitor_password';
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA tda_core, cybersecurity, finance, monitoring TO tda_api_user, tda_worker_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA tda_core, cybersecurity, finance, monitoring TO tda_api_user, tda_worker_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA tda_core, cybersecurity, finance, monitoring TO tda_api_user, tda_worker_user;

-- Monitor user gets read-only access
GRANT USAGE ON SCHEMA tda_core, cybersecurity, finance, monitoring TO tda_monitor_user;
GRANT SELECT ON ALL TABLES IN SCHEMA tda_core, cybersecurity, finance, monitoring TO tda_monitor_user;

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA tda_core, cybersecurity, finance, monitoring 
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO tda_api_user, tda_worker_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA tda_core, cybersecurity, finance, monitoring 
    GRANT SELECT ON TABLES TO tda_monitor_user;