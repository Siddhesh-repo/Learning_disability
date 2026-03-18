import React, { useState, useRef } from 'react';
import {
  Card, CardContent, Typography, Button, Box, Alert, CircularProgress,
  TextField, Table, TableBody, TableRow, TableCell,
} from '@mui/material';
import { Mic, Stop, CheckCircle } from '@mui/icons-material';
import { analyzeSpeech } from '../utils/api';

export default function SpeechTest({ onResult, result }) {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refText, setRefText] = useState(
    'The quick brown fox jumps over the lazy dog near the river bank.'
  );
  const mediaRef = useRef(null);
  const chunks = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunks.current = [];
      recorder.ondataavailable = (e) => chunks.current.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
      };
      recorder.start();
      mediaRef.current = recorder;
      setRecording(true);
      setError(null);
    } catch {
      setError('Microphone access denied. Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    mediaRef.current?.stop();
    setRecording(false);
  };

  const handleAnalyze = async () => {
    if (!audioBlob) { setError('Record audio first.'); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeSpeech(audioBlob, refText);
      onResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Speech analysis failed.');
    }
    setLoading(false);
  };

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h5" gutterBottom>Speech Analysis</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Ask the child to read the reference text below aloud while recording.
        </Typography>

        <TextField
          label="Reference Text"
          fullWidth multiline rows={2}
          value={refText}
          onChange={(e) => setRefText(e.target.value)}
          sx={{ mb: 2 }}
        />

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          {!recording ? (
            <Button variant="outlined" color="error" startIcon={<Mic />}
              onClick={startRecording}>
              Start Recording
            </Button>
          ) : (
            <Button variant="contained" color="error" startIcon={<Stop />}
              onClick={stopRecording}>
              Stop Recording
            </Button>
          )}

          <Button variant="contained" onClick={handleAnalyze}
            disabled={!audioBlob || loading || recording}>
            {loading ? <CircularProgress size={24} /> : 'Analyze Speech'}
          </Button>
        </Box>

        {audioBlob && !recording && (
          <Box sx={{ mb: 2 }}>
            <audio controls src={URL.createObjectURL(audioBlob)} />
          </Box>
        )}

        {result && (
          <Box>
            <Alert severity="success" icon={<CheckCircle />} sx={{ mb: 2 }}>
              Analysis complete — Confidence: <strong>{result.overall_score}</strong>
            </Alert>

            {result.quality?.warnings?.length > 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                {result.quality.warnings.join(' ')}
              </Alert>
            )}

            {result.transcript && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <strong>Transcript:</strong> {result.transcript}
              </Alert>
            )}

            <Table size="small">
              <TableBody>
                {Object.entries(result.features).map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell sx={{ textTransform: 'capitalize' }}>
                      {k.replace(/_/g, ' ')}
                    </TableCell>
                    <TableCell align="right">
                      {typeof v === 'number' ? v.toFixed(2) : String(v)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
