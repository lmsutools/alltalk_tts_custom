import React, { useEffect, useRef, useState } from 'react';
import './App.css';

const App = () => {
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(null);
  const [serverUrl, setServerUrl] = useState('https://511b46078d831.notebooksd.jarvislabs.net');
  const [inputText, setInputText] = useState('So If you encounter permission errors while installing packages, you can try running PowerShells. On Wikipedia and other sites running on MediaWiki');
  const [firstChunkReceivedAt, setFirstChunkReceivedAt] = useState(null);

  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const fetchAudioChunks = async () => {
    const startTime = performance.now();
    const url = new URL(`${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming`);
    url.searchParams.append('text', inputText);
    url.searchParams.append('voice', 'female_01.wav');
    url.searchParams.append('language', 'en');
    url.searchParams.append('output_file', 'output.mp3');
  
    const response = await fetch(url);
    const reader = response.body.getReader();
  
    const audioContext = audioContextRef.current;
    const audioSource = audioContext.createBufferSource();
    audioSource.connect(audioContext.destination);
  
    let isFirstChunk = true;
  
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
  
      if (isFirstChunk) {
        const firstChunkTime = (performance.now() - startTime) / 1000;
        setFirstChunkReceivedAt(firstChunkTime.toFixed(3));
        isFirstChunk = false;
      }
  
      const audioData = await audioContext.decodeAudioData(value.buffer);
      audioSource.buffer = audioData;
      audioSource.start();
    }
  
    const endTime = performance.now();
    setTotalResponseTime((endTime - startTime) / 1000);
    setTotalCharacterCount(inputText.length);
  };
  
  // Handlers for input fields changes
  const handleServerUrlChange = (e) => {
    setServerUrl(e.target.value);
  };
  
  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };
  

  return (
    <div className="app">
      <h1>Text-to-Speech Streaming App</h1>
      <div className="input-group">
        <label htmlFor="serverUrl">Server URL:</label>
        <input
          type="text"
          id="serverUrl"
          value={serverUrl}
          onChange={handleServerUrlChange}
          className="input-field"
        />
      </div>
      <div className="input-group">
        <label htmlFor="inputText">Input Text:</label>
        <textarea
          id="inputText"
          value={inputText}
          onChange={handleInputChange}
          rows={4}
          className="input-field"
        />
      </div>
      <button className="send-button" onClick={fetchAudioChunks}>
        Send Request
      </button>
      <div className="info-container">
        {firstChunkReceivedAt !== null && (
          <p>First Chunk Received At: {firstChunkReceivedAt} seconds</p>
        )}
        {totalCharacterCount !== null && (
          <p>Total Character Count: {totalCharacterCount}</p>
        )}
        {totalResponseTime !== null && (
          <p>Total Response Time: {totalResponseTime} seconds</p>
        )}
      </div>
    </div>
  );
};

export default App;