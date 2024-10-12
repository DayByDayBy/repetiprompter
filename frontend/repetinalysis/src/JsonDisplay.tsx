import React, { useState } from 'react';

interface JsonViewerProps {
  data: Record<string, any>;
}

const JsonViewer: React.FC<JsonViewerProps> = ({ data }) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggleExpand = (key: string) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const renderValue = (value: any, key: string) => {
    if (value === null) return <span className="ml-4">null</span>;
    if (typeof value === 'undefined') return <span className="ml-4">undefined</span>;

    if (typeof value === 'object') {
      return (
        <div className="ml-4">
          <span 
            className="cursor-pointer text-blue-500 hover:text-blue-700"
            onClick={() => toggleExpand(key)}
          >
            {expanded[key] ? '▼' : '▶'} {Array.isArray(value) ? `Array(${value.length})` : 'Object'}
          </span>
          {expanded[key] && (
            <div className="ml-4">
              <JsonViewer data={value} />
            </div>
          )}
        </div>
      );
    }

    return <span className="ml-4">{JSON.stringify(value)}</span>;
  };

  return (
    <div className="font-mono">
      {Object.entries(data).map(([key, value]) => (
        <div key={key}>
          <strong>{key}:</strong> {renderValue(value, key)}
        </div>
      ))}
    </div>
  );
};

export default JsonViewer;
