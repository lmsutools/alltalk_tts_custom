import React, { useState, useRef } from 'react';
import WavPlayer from 'webaudio-wav-stream-player';

const App = () => {
  const [serverUrl, setServerUrl] = useState('https://2ab7edc03be71.notebooksb.jarvislabs.net/');
  const [inputText, setInputText] = useState('So If you encounter permission errors while installing packages, you can try running PowerShells. On Wikipedia and other sites running on MediaWiki');
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(0);
  const [firstChunkReceivedAt, setFirstChunkReceivedAt] = useState(null);
  const [firstChunkPlayedAt, setFirstChunkPlayedAt] = useState(null);

  const player = useRef(new WavPlayer());
  const startTimeRef = useRef(null);

  const fetchAudioChunks = () => {
    const url = `${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming?text=${encodeURIComponent(inputText)}&voice=female_02.wav&language=en&output_file=output.wav`;
    setTotalCharacterCount(inputText.length);
    setFirstChunkReceivedAt(null);
    setFirstChunkPlayedAt(null);
    startTimeRef.current = performance.now();

    player.current.play(url).then(() => {
      setTotalResponseTime(performance.now() - startTimeRef.current);
    });

    player.current.onFirstChunkReceived = () => {
      setFirstChunkReceivedAt(performance.now() - startTimeRef.current);
    };

    player.current.onFirstChunkPlayed = () => {
      setFirstChunkPlayedAt(performance.now() - startTimeRef.current);
    };
  };

  const stopAudio = () => {
    player.current.stop();
  };

  return (
    <div>
      <h1>Audio Streaming Example</h1>
      <div>
        <label htmlFor="serverUrl">Server URL:</label>
        <input type="text" id="serverUrl" value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} />
      </div>
      <div>
        <label htmlFor="inputText">Input Text:</label>
        <textarea id="inputText" value={inputText} onChange={(e) => setInputText(e.target.value)}></textarea>
      </div>
      <button onClick={fetchAudioChunks}>Fetch Audio</button>
      <button onClick={stopAudio}>Stop Audio</button>
      <p>
        Total Response Time: {totalResponseTime !== null ? (totalResponseTime / 1000).toFixed(2) : '-'} seconds<br />
        Total Character Count: {totalCharacterCount}<br />

      </p>
    </div>
  );
};

export default App;