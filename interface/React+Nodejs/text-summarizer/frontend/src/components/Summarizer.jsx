import React, { useState } from 'react';
import axios from 'axios';
import './Summarizer.css'; // Import your custom CSS file

const Summarizer = () => {
    const [text, setText] = useState('');
    const [summary, setSummary] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSummarize = async () => {
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:5000/summarize', { text });
            setSummary(response.data.summary);
        } catch (error) {
            console.error('Error summarizing text:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <div className="header">
                <h1 className="title">Pegasus Text Summarizer</h1>
            </div>
            <div className="content">
                <div className="input-section">
                    <h2 className="section-title">Input Text</h2>
                    <textarea
                        className="input-text"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Paste your text here..."
                    ></textarea>
                    <button
                        className="summarize-button"
                        onClick={handleSummarize}
                        disabled={loading || !text.trim()}
                    >
                        {loading ? 'Summarizing...' : 'Summarize'}
                    </button>
                </div>
                <div className="output-section">
                    <h2 className="section-title">Summary</h2>
                    <div className="summary-text">
                        {summary ? (
                            <p>{summary}</p>
                        ) : (
                            <p className="placeholder-text">Summary will appear here</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Summarizer;
    