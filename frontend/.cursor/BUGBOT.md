# TDA Platform - Frontend Application BugBot Rules

## Frontend Architecture Overview
The frontend is a **React 18+ application** with TypeScript that provides an intuitive interface for TDA analysis workflows. It includes interactive visualizations using D3.js, Three.js, and Plotly, with real-time updates via WebSocket connections.

## Core Frontend Components

### 1. Application Structure
- **React 18+**: Functional components with hooks, modern patterns
- **TypeScript**: Strict mode, comprehensive type definitions
- **State Management**: React Query for server state, Context for UI state
- **Routing**: React Router with protected routes and authentication

### 2. Visualization Libraries
- **D3.js**: 2D visualizations (persistence diagrams, scatter plots)
- **Three.js**: 3D visualizations (point clouds, topological networks)
- **Plotly**: Interactive charts and graphs (Mapper algorithm output)
- **Canvas/WebGL**: High-performance rendering for large datasets

### 3. UI/UX Standards
- **Responsive Design**: Mobile-first approach, breakpoint system
- **Accessibility**: WCAG 2.1 AA compliance, screen reader support
- **Design System**: Consistent components, colors, typography, spacing
- **User Experience**: Intuitive workflows, clear feedback, error handling

## Frontend-Specific Rules

### 1. React Component Patterns
```typescript
// ✅ DO: Use functional components with hooks
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';

interface AnalysisConfigProps {
  onConfigChange: (config: AnalysisConfig) => void;
  initialConfig?: AnalysisConfig;
}

export const AnalysisConfig: React.FC<AnalysisConfigProps> = ({
  onConfigChange,
  initialConfig
}) => {
  const [config, setConfig] = useState<AnalysisConfig>(initialConfig || defaultConfig);
  
  const handleConfigChange = useCallback((newConfig: Partial<AnalysisConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    onConfigChange(updatedConfig);
  }, [config, onConfigChange]);
  
  return (
    <div className="analysis-config">
      {/* Component implementation */}
    </div>
  );
};

// ❌ DON'T: Use class components or inline functions
export class AnalysisConfig extends React.Component {  // Bad: class component
  render() {
    return (
      <button onClick={() => this.handleClick()}>  // Bad: inline function
        Click me
      </button>
    );
  }
}
```

### 2. TypeScript Best Practices
```typescript
// ✅ DO: Comprehensive type definitions
interface PersistenceDiagram {
  points: Array<{
    birth: number;
    death: number;
    dimension: number;
    multiplicity: number;
  }>;
  metadata: {
    filtrationType: 'vietoris_rips' | 'alpha_complex' | 'cech' | 'dtm';
    maxDimension: number;
    totalPoints: number;
  };
}

interface AnalysisRequest {
  name: string;
  points: number[][];
  filtrationType: PersistenceDiagram['metadata']['filtrationType'];
  maxDimension: number;
  parameters?: Record<string, any>;
}

// ❌ DON'T: Use any or loose typing
interface AnalysisRequest {  // Bad: loose typing
  name: any;
  points: any[];
  filtrationType: any;
}
```

### 3. State Management Patterns
```typescript
// ✅ DO: Use React Query for server state
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export const useAnalysis = (analysisId: string) => {
  return useQuery({
    queryKey: ['analysis', analysisId],
    queryFn: () => api.getAnalysis(analysisId),
    enabled: !!analysisId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useCreateAnalysis = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: AnalysisRequest) => api.createAnalysis(data),
    onSuccess: (newAnalysis) => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
      queryClient.setQueryData(['analysis', newAnalysis.id], newAnalysis);
    },
    onError: (error) => {
      console.error('Failed to create analysis:', error);
      // Handle error appropriately
    },
  });
};

// ❌ DON'T: Manage server state manually
const [analysis, setAnalysis] = useState<Analysis | null>(null);
const [loading, setLoading] = useState(false);

useEffect(() => {  // Bad: manual state management
  setLoading(true);
  api.getAnalysis(analysisId)
    .then(setAnalysis)
    .finally(() => setLoading(false));
}, [analysisId]);
```

### 4. API Integration Patterns
```typescript
// ✅ DO: Centralized API client with error handling
class ApiClient {
  private baseUrl: string;
  private authToken: string | null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.authToken = localStorage.getItem('auth_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.authToken) {
      headers.Authorization = `Bearer ${this.authToken}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new ApiError(
          response.status,
          response.statusText,
          await response.text()
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      throw new ApiError(0, 'Network error', error.message);
    }
  }

  async createAnalysis(data: AnalysisRequest): Promise<Analysis> {
    return this.request<Analysis>('/analyses', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

// ❌ DON'T: Scattered API calls without error handling
const createAnalysis = async (data: AnalysisRequest) => {
  const response = await fetch('/api/analyses', {  // Bad: no error handling
    method: 'POST',
    body: JSON.stringify(data),
  });
  return response.json();
};
```

## TDA-Specific Frontend Patterns

### 1. Data Upload Components
```typescript
// ✅ DO: Comprehensive file upload with validation
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileSelect: (file: File, metadata: FileMetadata) => void;
  acceptedFormats: string[];
  maxSize: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  acceptedFormats,
  maxSize
}) => {
  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    accept: acceptedFormats.reduce((acc, format) => ({
      ...acc,
      [format]: []
    }), {}),
    maxSize,
    multiple: false,
    onDropAccepted: (files) => {
      const file = files[0];
      const metadata = extractFileMetadata(file);
      onFileSelect(file, metadata);
    },
    onDropRejected: (rejections) => {
      const error = rejections[0].errors[0];
      if (error.code === 'file-too-large') {
        showError(`File size exceeds ${maxSize / (1024 * 1024)}MB limit`);
      } else if (error.code === 'file-invalid-type') {
        showError(`Invalid file type. Accepted formats: ${acceptedFormats.join(', ')}`);
      }
    }
  });

  return (
    <div {...getRootProps()} className={`file-upload ${isDragActive ? 'drag-active' : ''}`}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the file here...</p>
      ) : (
        <p>Drag & drop a file here, or click to select</p>
      )}
    </div>
  );
};

// ❌ DON'T: Basic file input without validation
<input type="file" onChange={(e) => onFileSelect(e.target.files[0])} />
```

### 2. Analysis Configuration Forms
```typescript
// ✅ DO: Form validation with proper error handling
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const analysisConfigSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100, 'Name too long'),
  filtrationType: z.enum(['vietoris_rips', 'alpha_complex', 'cech', 'dtm']),
  maxDimension: z.number().min(0).max(3),
  parameters: z.object({
    epsilon: z.number().positive().optional(),
    alpha: z.number().positive().optional(),
  }).optional(),
});

type AnalysisConfigForm = z.infer<typeof analysisConfigSchema>;

export const AnalysisConfigForm: React.FC<{
  onSubmit: (config: AnalysisConfigForm) => void;
  initialValues?: Partial<AnalysisConfigForm>;
}> = ({ onSubmit, initialValues }) => {
  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
  } = useForm<AnalysisConfigForm>({
    resolver: zodResolver(analysisConfigSchema),
    defaultValues: initialValues,
  });

  const filtrationType = watch('filtrationType');

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="analysis-config-form">
      <div className="form-group">
        <label htmlFor="name">Analysis Name</label>
        <Controller
          name="name"
          control={control}
          render={({ field }) => (
            <input
              {...field}
              id="name"
              type="text"
              className={errors.name ? 'error' : ''}
            />
          )}
        />
        {errors.name && <span className="error-message">{errors.name.message}</span>}
      </div>

      <div className="form-group">
        <label htmlFor="filtrationType">Filtration Type</label>
        <Controller
          name="filtrationType"
          control={control}
          render={({ field }) => (
            <select {...field} id="filtrationType">
              <option value="vietoris_rips">Vietoris-Rips</option>
              <option value="alpha_complex">Alpha Complex</option>
              <option value="cech">Čech Complex</option>
              <option value="dtm">Distance-to-Measure</option>
            </select>
          )}
        />
      </div>

      {filtrationType === 'vietoris_rips' && (
        <div className="form-group">
          <label htmlFor="epsilon">Epsilon Value</label>
          <Controller
            name="parameters.epsilon"
            control={control}
            render={({ field }) => (
              <input
                {...field}
                id="epsilon"
                type="number"
                step="0.01"
                min="0"
              />
            )}
          />
        </div>
      )}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Creating...' : 'Create Analysis'}
      </button>
    </form>
  );
};
```

### 3. Visualization Components
```typescript
// ✅ DO: Responsive, interactive visualizations
import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';

interface PersistenceDiagramProps {
  data: PersistenceDiagram;
  width?: number;
  height?: number;
  onPointClick?: (point: PersistencePoint) => void;
}

export const PersistenceDiagram: React.FC<PersistenceDiagramProps> = ({
  data,
  width = 600,
  height = 400,
  onPointClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Responsive sizing
  const [dimensions, setDimensions] = useState({ width, height });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth;
        setDimensions({
          width: Math.min(containerWidth - 40, width),
          height: Math.min(containerWidth * 0.6, height)
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [width, height]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || !data.points.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous content

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data.points, d => d.death) || 1])
      .range([0, chartWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data.points, d => d.death) || 1])
      .range([chartHeight, 0]);

    // Color scale for dimensions
    const colorScale = d3.scaleOrdinal()
      .domain([0, 1, 2, 3])
      .range(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']);

    // Add chart group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));

    // Add diagonal line (birth = death)
    g.append('line')
      .attr('x1', 0)
      .attr('y1', chartHeight)
      .attr('x2', chartWidth)
      .attr('y2', 0)
      .attr('stroke', '#ccc')
      .attr('stroke-dasharray', '5,5');

    // Add points
    g.selectAll('circle')
      .data(data.points)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.birth))
      .attr('cy', d => yScale(d.death))
      .attr('r', 4)
      .attr('fill', d => colorScale(d.dimension))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('click', (event, d) => onPointClick?.(d))
      .append('title')
      .text(d => `Dimension ${d.dimension}: (${d.birth.toFixed(3)}, ${d.death.toFixed(3)})`);

  }, [data, dimensions, onPointClick]);

  return (
    <div ref={containerRef} className="persistence-diagram-container">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="persistence-diagram"
      />
    </div>
  );
};

// ❌ DON'T: Static, non-responsive visualizations
export const PersistenceDiagram: React.FC<{ data: PersistenceDiagram }> = ({ data }) => {
  return (
    <svg width="600" height="400">  // Bad: fixed dimensions
      {/* Static visualization */}
    </svg>
  );
};
```

### 4. Real-time Updates
```typescript
// ✅ DO: WebSocket integration with proper error handling
import { useEffect, useRef, useCallback } from 'react';

interface WebSocketHookOptions {
  url: string;
  onMessage: (data: any) => void;
  onError?: (error: Event) => void;
  onClose?: () => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  onClose,
  reconnectAttempts = 5,
  reconnectInterval = 3000
}: WebSocketHookOptions) => {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectCountRef = useRef(0);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        onClose?.();

        // Attempt reconnection
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++;
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }, [url, onMessage, onError, onClose, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { sendMessage, disconnect };
};

// ❌ DON'T: Basic WebSocket without error handling
useEffect(() => {
  const ws = new WebSocket(url);  // Bad: no error handling
  ws.onmessage = onMessage;
  return () => ws.close();
}, [url, onMessage]);
```

## Accessibility & UX

### 1. WCAG 2.1 AA Compliance
```typescript
// ✅ DO: Implement proper accessibility features
export const AccessibleButton: React.FC<{
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
  ariaLabel?: string;
}> = ({ onClick, children, disabled, ariaLabel }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-disabled={disabled}
      className={`btn ${disabled ? 'btn--disabled' : ''}`}
    >
      {children}
    </button>
  );
};

// ❌ DON'T: Missing accessibility attributes
<button onClick={onClick}>{children}</button>  // Bad: no accessibility
```

### 2. Error Handling & User Feedback
```typescript
// ✅ DO: Provide clear error messages and loading states
export const AnalysisStatus: React.FC<{ analysisId: string }> = ({ analysisId }) => {
  const { data: analysis, error, isLoading } = useAnalysis(analysisId);

  if (isLoading) {
    return (
      <div className="analysis-status loading" role="status" aria-live="polite">
        <div className="spinner" aria-label="Loading analysis status" />
        <span>Loading analysis status...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analysis-status error" role="alert">
        <h3>Failed to load analysis</h3>
        <p>{error.message}</p>
        <button onClick={() => window.location.reload()}>
          Try again
        </button>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="analysis-status not-found">
        <p>Analysis not found</p>
      </div>
    );
  }

  return (
    <div className="analysis-status">
      <h3>Analysis Status: {analysis.name}</h3>
      <div className="status-indicator">
        <span className={`status-badge status-${analysis.status}`}>
          {analysis.status}
        </span>
      </div>
      {analysis.progress && (
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${analysis.progress}%` }}
            role="progressbar"
            aria-valuenow={analysis.progress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      )}
    </div>
  );
};
```

## Testing Frontend Components

### 1. Component Testing
```typescript
// ✅ DO: Comprehensive component testing
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AnalysisConfig } from './AnalysisConfig';

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <QueryClientProvider client={createTestQueryClient()}>
    {children}
  </QueryClientProvider>
);

describe('AnalysisConfig', () => {
  it('renders form fields correctly', () => {
    const mockOnConfigChange = jest.fn();
    
    render(
      <TestWrapper>
        <AnalysisConfig onConfigChange={mockOnConfigChange} />
      </TestWrapper>
    );

    expect(screen.getByLabelText(/analysis name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/filtration type/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max dimension/i)).toBeInTheDocument();
  });

  it('calls onConfigChange when form is submitted', async () => {
    const mockOnConfigChange = jest.fn();
    
    render(
      <TestWrapper>
        <AnalysisConfig onConfigChange={mockOnConfigChange} />
      </TestWrapper>
    );

    fireEvent.change(screen.getByLabelText(/analysis name/i), {
      target: { value: 'Test Analysis' }
    });

    fireEvent.click(screen.getByRole('button', { name: /create analysis/i }));

    await waitFor(() => {
      expect(mockOnConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Test Analysis'
        })
      );
    });
  });

  it('shows validation errors for invalid input', async () => {
    const mockOnConfigChange = jest.fn();
    
    render(
      <TestWrapper>
        <AnalysisConfig onConfigChange={mockOnConfigChange} />
      </TestWrapper>
    );

    // Submit without filling required fields
    fireEvent.click(screen.getByRole('button', { name: /create analysis/i }));

    await waitFor(() => {
      expect(screen.getByText(/name is required/i)).toBeInTheDocument();
    });
  });
});
```

## Common Frontend Issues

### 1. Performance Issues
- **Large Re-renders**: Use React.memo, useMemo, useCallback
- **Memory Leaks**: Clean up event listeners, timeouts, subscriptions
- **Bundle Size**: Code splitting, lazy loading, tree shaking

### 2. State Management
- **Prop Drilling**: Use Context API or state management libraries
- **Stale State**: Proper dependency arrays in useEffect, useCallback
- **Race Conditions**: Cancel previous requests, use AbortController

### 3. Accessibility Issues
- **Missing ARIA Labels**: Provide proper labels for interactive elements
- **Keyboard Navigation**: Ensure all functionality is keyboard accessible
- **Screen Reader Support**: Use semantic HTML, proper heading structure

### 4. Error Boundaries
```typescript
// ✅ DO: Implement error boundaries for graceful error handling
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    // Log to error reporting service
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary" role="alert">
          <h2>Something went wrong</h2>
          <p>We're sorry, but something unexpected happened.</p>
          <button onClick={() => window.location.reload()}>
            Refresh the page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```
