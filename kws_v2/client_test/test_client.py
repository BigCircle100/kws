import asyncio
import websockets
import wave
import json
import time
from scipy.io.wavfile import write
from scipy.signal import resample
import numpy as np


async def send_audio_file(file_path, uri):
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")


        start_data = {
            "messageType": "inference",
            "param": {
                "format": "INT16",
                "rate": 44100,
                "time_per_buffer": 200,
                "channels": 1
            }
        }

        json_data = json.dumps(start_data)

        await websocket.send(json_data)

        
        idx = 0
        target_rate = 8000

        total_data = np.array([], dtype=np.int16)


        with wave.open(file_path, 'rb') as wf:
            frame_rate = wf.getframerate()
            chunk_size = int(frame_rate * 0.1)  
            print(f"Frame rate: {frame_rate}, Chunk size: {chunk_size}")

            while True:
                audio_data = wf.readframes(chunk_size)
                if not audio_data:
                    print("End of audio file reached.")
                    break  

                new_data = np.frombuffer(audio_data, dtype=np.int16)
                print(new_data.shape)
                target_num_samples = int(len(new_data)*target_rate/frame_rate)
                new_data = resample(new_data, target_num_samples).astype(np.int16)

                total_data = np.concatenate((total_data, new_data))
                if (len(total_data) >= 16000):
                    idx += 1
                    print(idx)
                    total_data = total_data[len(total_data)-16000:]
                    # total_data.tofile(f"client_cut/{idx}.bin")
                    # write(f"client_cut/{idx}.wav", 8000, total_data)

                # await websocket.send(total_data.tobytes())
                await websocket.send(audio_data)
                print("Sent a chunk of audio data.")

                await asyncio.sleep(0.2)

async def main():
    uri = "ws://localhost:18080/inference" 
    file_path = "out(2).wav"  
    await send_audio_file(file_path, uri)

if __name__ == "__main__":
    asyncio.run(main())