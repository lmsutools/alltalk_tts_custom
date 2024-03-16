import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const audioRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(null);
  const [serverUrl, setServerUrl] = useState('https://f4aacd6da9151.notebooksb.jarvislabs.net');
  const [inputText, setInputText] = useState('So If you encounter permission errors while installing packages, you can try running PowerShells. On Wikipedia and other sites running on MediaWiki');
  const [firstChunkReceivedAt, setFirstChunkReceivedAt] = useState(null);

  const fetchAudioChunks = async () => {
    const startTime = performance.now();
    const response = await axios.get(`${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming`, {
      params: {
        text: inputText,
        voice: 'female_01.wav',
        language: 'en',
      },
      responseType: 'stream',
    });

    const mediaSource = new MediaSource();
    audioRef.current.src = URL.createObjectURL(mediaSource);
    mediaSourceRef.current = mediaSource;

    mediaSource.addEventListener('sourceopen', () => {
      const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');

      const onUpdateEnd = () => {
        if (mediaSource.readyState === 'open') {
          mediaSource.endOfStream();
        }
      };

      sourceBuffer.addEventListener('updateend', onUpdateEnd);

      let isFirstChunk = true;

      response.data.on('data', (chunk) => {
        if (isFirstChunk) {
          const firstChunkTime = (performance.now() - startTime) / 1000;
          setFirstChunkReceivedAt(firstChunkTime.toFixed(3));
          isFirstChunk = false;
        }

        sourceBuffer.appendBuffer(new Uint8Array(chunk));
      });

      response.data.on('end', () => {
        const endTime = performance.now();
        setTotalResponseTime((endTime - startTime) / 1000);
        setTotalCharacterCount(inputText.length);
      });
    });
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleServerUrlChange = (event) => {
    setServerUrl(event.target.value);
  };

  return (
    <div className="app">
      <h1>Text-to-Speech Streaming App</h1>
      <div className="input-group">
        <label htmlFor="serverUrl">Server URL:</label>
        <input type="text" id="serverUrl" value={serverUrl} onChange={handleServerUrlChange} className="input-field" />
      </div>
      <div className="input-group">
        <label htmlFor="inputText">Input Text:</label>
        <textarea id="inputText" value={inputText} onChange={handleInputChange} rows={4} className="input-field" />
      </div>
      <button className="send-button" onClick={fetchAudioChunks}>Send Request</button>
      <audio ref={audioRef} autoPlay />
      <div className="info-container">
        {firstChunkReceivedAt !== null && <p>First Chunk Received At: {firstChunkReceivedAt} seconds</p>}
        {totalCharacterCount !== null && <p>Total Character Count: {totalCharacterCount}</p>}
        {totalResponseTime !== null && <p>Total Response Time: {totalResponseTime} seconds</p>}
      </div>
    </div>
  );
};

export default App;