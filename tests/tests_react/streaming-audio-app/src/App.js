
import React, { useEffect, useRef, useState } from 'react';
import './App.css';

const App = () => {
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(null);
  const [serverUrl, setServerUrl] = useState('https://d0987c0201e11.notebooksi.jarvislabs.net');
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
    const url = `${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming?text=${encodeURIComponent(inputText)}&voice=female_01.wav&language=en&output_file=output.mp3`;
    const response = await fetch(url);
    const reader = response.body.getReader();

    const mediaSource = new MediaSource();
    const audio = new Audio();
    audio.src = URL.createObjectURL(mediaSource);

    mediaSource.addEventListener('sourceopen', async () => {
      try {
        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg'); // Ensure this matches the MIME type of your audio
        audio.play(); // Attempt to play audio, though actual playback may start after appending data

        let isFirstChunk = true;
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            mediaSource.endOfStream();
            const endTime = performance.now();
            setTotalResponseTime(((endTime - startTime) / 1000).toFixed(3));
            setTotalCharacterCount(inputText.length);
            break;
          }
          if (isFirstChunk) {
            setFirstChunkReceivedAt(((performance.now() - startTime) / 1000).toFixed(3));
            isFirstChunk = false;
          }
          if (sourceBuffer.updating) {
            await new Promise(resolve => sourceBuffer.addEventListener('updateend', resolve, { once: true }));
          }
          sourceBuffer.appendBuffer(value);
        }
      } catch (error) {
        console.error('Error during fetch and playback:', error);
      }
    });
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