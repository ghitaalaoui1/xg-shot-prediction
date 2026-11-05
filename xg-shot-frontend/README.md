# PitchIQ — xG Shot Prediction frontend

React + Vite frontend for the existing FastAPI backend in `ghitaalaoui1/xg-shot-prediction`.

## Run locally

1. Keep your existing Python backend unchanged.
2. Start FastAPI from the repository root:

   ```bash
   uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
   ```

   If your current working directory/module invocation differs, use the same command you already use to launch the API; the important part is that it listens on `127.0.0.1:8000`.

3. In another terminal:

   ```bash
   cd xg-shot-frontend
   npm install
   npm run dev
   ```

4. Open the Vite URL printed in the terminal (usually `http://localhost:5173`).

Vite proxies `/api/predict` to `http://127.0.0.1:8000/predict`, so no CORS change is needed for local development.

## Backend contract

The frontend posts the 24 fields required by `api/main.py` to `POST /predict` and reads `probabilite_de_but` from the JSON response. The frontend computes only the geometric input values (distance and shot angle) from the selected pitch location; the prediction itself always comes from the Python model.

## Notes

- The visual pitch uses the StatsBomb-style 120 × 80 coordinate space and the attacking half (x = 60–120).
- Defender count and goalkeeper measurements are scenario inputs. The frontend does not infer player positions from a freeze frame.
- The model notes page uses project details from the repository README; it does not claim to fetch live evaluation metrics.
