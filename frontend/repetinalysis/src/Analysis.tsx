import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const AnalysisDisplay = ({ fileId }) => {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch(`/api/analysis/${fileId}`)
            .then(response => response.json())
            .then(setData);
    }, [fileId]);

    if (!data) return <div>Loading...</div>;

    return (
        <div className= "h-64 w-full" >
        <ResponsiveContainer>
        <BarChart data={ data.wordFrequencies }>
            <XAxis dataKey="word" />
                <YAxis />
                < Tooltip />
                <Bar dataKey="frequency" fill = "#8884d8" />
                    </BarChart>
                    </ResponsiveContainer>
                    </div>
  );
};

export default AnalysisDisplay;