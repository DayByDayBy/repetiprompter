import React, { useState } from 'react';

interface JsonViewerProps {
    data: Record<string, any>;
    maxDepth?: number;
  }

  
  const JsonViewer: React.FC<JsonViewerProps> = ({ data, maxDepth = 10 }) => {
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggleExpand = (key: string) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const renderArray = React.useCallback((arr: any[], key: string) => {
    return (
      <div className="ml-4">
        <span 
          className="cursor-pointer text-blue-500 hover:text-blue-700"
          onClick={() => toggleExpand(key)}
        >
          {expanded[key] ? `▼ Array(${arr.length})` : `▶ Array(${arr.length})`}
        </span>
        {expanded[key] && (
          <div className="ml-4">
            {arr.map((item, index) => (
              <div key={`${key}-${index}`}>
                {typeof item === 'object' ? <JsonViewer data={item} /> : JSON.stringify(item)}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }, [expanded, toggleExpand]);
  ;

  const renderValue = React.useCallback((value: any, key: string, depth: number) => {
    if (depth > maxDepth) return <span className="ml-4">Max depth reached</span>;
    if (typeof value === 'undefined') return <span className="ml-4">undefined</span>;

    if (Array.isArray(value)) {
      return renderArray(value, key); // render array
    }

    if (typeof value === 'object') {
      return (
        <div className="ml-4">
          <span 
            className="cursor-pointer text-blue-500 hover:text-blue-700"
            onClick={() => toggleExpand(key)}
          >
            {expanded[key] ? '▼ Object' : '▶ Object'}
          </span>
          {expanded[key] && (
            <div className="ml-4">
              <JsonViewer data={value} />
            </div>
          )}
        </div>
      );
    }

    // render scalar values directly:
    if (typeof value === 'number' && !Number.isInteger(value)) {
      return <span className="ml-4">{value.toFixed(2)}</span>; // 2 decimal places, seems easier
    }
    return <span className="ml-4">{String(value)}</span>;


  }, [expanded, toggleExpand, renderArray, maxDepth]);
  ;

  const renderMetadataValue = (value: any): string => {
    if (typeof value === 'object' && value !== null) {
      if (Array.isArray(value)) {
        return `[Array(${value.length})]`;
      } else {
        return '{Object}';
      }
    }
    return String(value);
  };
  
  const renderMetadata = React.useCallback((metadata: Record<string, any>) => {
    return (
      <div className="metadata-section bg-gray-100 p-4 rounded">
        <h3 className="text-lg font-bold mb-2">Metadata</h3>
        <ul className="space-y-1">
          {Object.entries(metadata).map(([key, value]) => (
            <li key={key} className="flex">
              <strong className="mr-2">{key}:</strong>
              {renderValue(value, key, 0)}
            </li>
          ))}
        </ul>
      </div>
    );
  }, [renderValue]);

  return (
    <div className="font-mono">
      {/* check if there's metadata, render it separately */}
      {data.metadata && renderMetadata(data.metadata)}

      {/* render the rest of the JSON data */}
      {Object.entries(data).filter(([key]) => key !== 'metadata').map(([key, value]) => (
        <div key={key}>
          <strong>{key}:</strong> {renderValue(value, key)}
        </div>
      ))}
    </div>
  );
};

export default JsonViewer;
