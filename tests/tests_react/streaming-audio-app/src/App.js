import React, { useEffect, useRef, useState } from 'react';
import './App.css';

const App = () => {
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(null);
  const [serverUrl, setServerUrl] = useState('https://bbe8dd017c361.notebooksc.jarvislabs.net');
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
    url.searchParams.append('output_file', 'output.wav');
  
    const response = await fetch(url);
    const reader = response.body.getReader();
  
    const audioContext = audioContextRef.current;
    const sampleRate = 24000;
    let currentTime = 0;
    let isFirstChunk = true;
    let remainingBytes = new Uint8Array(0);
  
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
  
      if (isFirstChunk) {
        const firstChunkTime = (performance.now() - startTime) / 1000;
        setFirstChunkReceivedAt(firstChunkTime.toFixed(3));
        isFirstChunk = false;
      }
  
      const combinedBytes = new Uint8Array(remainingBytes.length + value.length);
      combinedBytes.set(remainingBytes);
      combinedBytes.set(value, remainingBytes.length);
  
      const bytesPerSample = 4;
      const numSamples = Math.floor(combinedBytes.length / bytesPerSample);
      const sampleBytes = combinedBytes.slice(0, numSamples * bytesPerSample);
      remainingBytes = combinedBytes.slice(numSamples * bytesPerSample);
  
      const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
      const float32Array = new Float32Array(sampleBytes.buffer);
      audioBuffer.copyToChannel(float32Array, 0, 0);
  
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start(currentTime);
  
      currentTime += audioBuffer.duration;
    }
  
    const endTime = performance.now();
    setTotalResponseTime((endTime - startTime) / 1000);
    setTotalCharacterCount(inputText.length);
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