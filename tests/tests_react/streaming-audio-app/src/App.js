import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const audioRef = useRef(null);
  const [audioInfo, setAudioInfo] = useState(null);
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
        output_file: 'output.wav',
      },
      responseType: 'arraybuffer',
    });
  
    const audioContext = new AudioContext();
    
    let isFirstChunk = true;
    const processChunk = async (chunk) => {
      try {
        const audioBuffer = await audioContext.decodeAudioData(chunk);
        if (isFirstChunk) {
          setAudioInfo({
            sampleRate: audioContext.sampleRate,
            numberOfChannels: audioBuffer.numberOfChannels,
          });
  
          const firstChunkTime = (performance.now() - startTime) / 1000;
          setFirstChunkReceivedAt(firstChunkTime.toFixed(3));
          isFirstChunk = false;
        }
  
        // Play the audioBuffer here
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
        // You need to handle the end of playback to start the next chunk smoothly
      } catch (error) {
        console.error('Error decoding audio data:', error);
      }
    };
  
    let offset = 0;
    const chunkSize = 4096;
    const chunks = []; // Assuming you might want to manage chunks for sequential processing
  
    while (offset < response.data.byteLength) {
      const chunk = response.data.slice(offset, offset + chunkSize);
      chunks.push(chunk);
      offset += chunkSize;
    }
  
    for (const chunk of chunks) {
      await processChunk(chunk);
      // Wait for the chunk to finish playing or use a more sophisticated method to ensure smooth playback
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
      <button className="send-button" onClick={fetchAudioChunks}>Send Request</button>
      <audio ref={audioRef} />
      <div className="info-container">
        {audioInfo && (
          <div>
            <h3>Audio Information:</h3>
            <p>Sample Rate: {audioInfo.sampleRate} Hz</p>
            <p>Number of Channels: {audioInfo.numberOfChannels}</p>
          </div>
        )}
        {firstChunkReceivedAt !== null && <p>First Chunk Received At: {firstChunkReceivedAt} seconds</p>}
        {totalCharacterCount !== null && <p>Total Character Count: {totalCharacterCount}</p>}
        {totalResponseTime !== null && <p>Total Response Time: {totalResponseTime} seconds</p>}
      </div>
    </div>
  );
};

export default App;