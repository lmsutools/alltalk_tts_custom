import React, { useState, useEffect, useRef } from 'react';
import WavPlayer from 'webaudio-wav-stream-player';

const App = () => {
  const [serverUrl, setServerUrl] = useState('https://4c10664666e31.notebooksj.jarvislabs.net/');
  const [inputText, setInputText] = useState('So If you encounter permission errors while installing packages, you can try running PowerShell. On Wikipedia and other sites running on MediaWiki');
  const [totalResponseTime, setTotalResponseTime] = useState(null);
  const [totalCharacterCount, setTotalCharacterCount] = useState(0);
  const [isParallelActive, setIsParallelActive] = useState(false);
  const [rampUp, setRampUp] = useState(1);
  const [numberOfUsers, setNumberOfUsers] = useState(1);
  const [userResults, setUserResults] = useState([]);

  const player = useRef(new WavPlayer());
  const startTimeRef = useRef(null);

  useEffect(() => {
    // Set event listeners for the player
    player.current.onFirstChunkReceived = () => {
      console.log("First chunk received");
    };
    player.current.onFirstChunkPlayed = () => {
      console.log("First chunk played");
    };
  }, []);

  const fetchAudioChunks = async () => {
    setUserResults([]); // Reset results before starting new tests
    setTotalCharacterCount(inputText.length);
    startTimeRef.current = performance.now();

    if (isParallelActive) {
      let activeRequests = 0;
      for (let i = 0; i < numberOfUsers; i++) {
        setTimeout(async () => {
          activeRequests++;
          const playerInstance = new WavPlayer();
          const startTime = performance.now();
          await playerInstance.play(`${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming?text=${encodeURIComponent(inputText)}&voice=female_02.wav&language=en&output_file=output.wav`);
          const latency = performance.now() - startTime;
          setUserResults((currentResults) => [
            ...currentResults,
            { user: i + 1, latency: latency.toFixed(2) }
          ]);
          activeRequests--;
          if (activeRequests === 0) {
            setTotalResponseTime(performance.now() - startTimeRef.current);
          }
        }, i * rampUp * 1000);
      }
    } else {
      await player.current.play(`${serverUrl.replace(/\/$/, '')}/api/tts-generate-streaming?text=${encodeURIComponent(inputText)}&voice=female_02.wav&language=en&output_file=output.wav`);
      setTotalResponseTime(performance.now() - startTimeRef.current);
    }
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
      <div>
        <label>
          <input type="checkbox" checked={isParallelActive} onChange={(e) => setIsParallelActive(e.target.checked)} />
          Parallel Test Active
        </label>
      </div>
      {isParallelActive && (
        <>
          <div>
            <label htmlFor="rampUp">Ramp Up (seconds):</label>
            <input type="number" id="rampUp" value={rampUp} onChange={(e) => setRampUp(parseInt(e.target.value, 10))} />
          </div>
          <div>
            <label htmlFor="numberOfUsers">Number of Users:</label>
            <input type="number" id="numberOfUsers" value={numberOfUsers} onChange={(e) => setNumberOfUsers(parseInt(e.target.value, 10))} />
          </div>
        </>
      )}
      <button onClick={fetchAudioChunks}>Fetch Audio</button>
      <button onClick={stopAudio}>Stop Audio</button>
      <p>
        Total Response Time: {totalResponseTime !== null ?
(totalResponseTime / 1000).toFixed(2) : '-'} seconds<br />
Total Character Count: {totalCharacterCount}<br />
</p>
{userResults.length > 0 && (
<div>
  <h2>Parallel Test Results:</h2>
  {userResults.map((result, index) => (
    <p key={index}>
      USER {result.user}: Latency: {result.latency}ms
    </p>
  ))}
</div>
)}
</div>
);
};

export default App;
