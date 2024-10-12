import React, { useState, useEffect } from 'react';
import JsonViewer from './components/JsonDisplay';

// dynamically importing JSON files from the 'responses' folder
const importAllJson = (requireContext: __WebpackModuleApi.RequireContext) => {
  let jsonFiles: Record<string, any> = {};
  requireContext.keys().forEach((file: string) => {
    const fileName = file.replace('./', ''); // Remove the leading './' from the filename
    jsonFiles[fileName] = requireContext(file); // Load the JSON file
  });
  return jsonFiles;
};

const App: React.FC = () => {
  const [jsonFiles, setJsonFiles] = useState<Record<string, any>>({});
  const [currentFile, setCurrentFile] = useState<string>(''); // State for selected JSON file
  const [jsonData, setJsonData] = useState<any>(null);

  useEffect(() => {
    // Use require.context to dynamically load all JSON files from the 'responses' folder
    const files = importAllJson(require.context('../../../../repetiprompter/responses', false, /\.json$/));
    setJsonFiles(files); // Set the loaded files in state
    setCurrentFile(Object.keys(files)[0]); // Set the first file as the default selected file
  }, []);

  useEffect(() => {
    if (currentFile && jsonFiles[currentFile]) {
      setJsonData(jsonFiles[currentFile]); // update jsonData when  currentFile changes
    }
  }, [currentFile, jsonFiles]);

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="text-2xl font-bold mb-4">JSON Viewer</h1>

        {/* to select a JSON file */}
        <select 
          value={currentFile} 
          onChange={(e) => setCurrentFile(e.target.value)}
          className="mb-4 p-2 border rounded"
        >
          {Object.keys(jsonFiles).map((fileName) => (
            <option key={fileName} value={fileName}>
              {fileName}
            </option>
          ))}
        </select>

        {/* Render JsonViewer if jsonData is available */}
        {jsonData ? (
          <JsonViewer data={jsonData} maxDepth={10} />
        ) : (
          <p>Loading JSON data...</p>
        )}
      </header>
    </div>
  );
};

export default App;
