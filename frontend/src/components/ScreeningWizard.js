import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box, Container, Typography, Button, Alert, LinearProgress, Paper
} from '@mui/material';
import { NavigateNext, NavigateBefore } from '@mui/icons-material';

import StepIndicator from './StepIndicator';
import ConsentScreen from './ConsentScreen';
import HandwritingTest from './HandwritingTest';
import SpeechTest from './SpeechTest';
import AssessmentReport from './AssessmentReport';
import {
  predict,
  predictFusionPhase4,
  predictSpeechPhase3,
  predictHandwritingPhase2,
  getStudentById
} from '../utils/api';

const STEPS = [
  'Consent & Disclaimer',
  'Handwriting Analysis',
  'Speech Analysis',
  'Assessment Report',
];

export default function ScreeningWizard() {
  const { studentId } = useParams();
  const navigate = useNavigate();
  
  const [student, setStudent] = useState(null);
  const [initLoading, setInitLoading] = useState(true);

  const [step, setStep] = useState(0);
  const [consent, setConsent] = useState(false);
  
  const [hwResult, setHwResult] = useState(null);
  const [spResult, setSpResult] = useState(null);
  const [prediction, setPrediction] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStudent = async () => {
      try {
        const res = await getStudentById(studentId);
        setStudent(res.data.student);
      } catch (err) {
        setError('Failed to fetch student profile.');
      } finally {
        setInitLoading(false);
      }
    };
    fetchStudent();
  }, [studentId]);

  const canAdvance = () => {
    if (step === 0 && !consent) return 'Please accept the consent to continue.';
    if (step === 1 && !hwResult) return 'Complete handwriting analysis first.';
    if (step === 2 && !spResult) return 'Complete speech analysis first.';
    return null;
  };

  const handleNext = async () => {
    const msg = canAdvance();
    if (msg) { setError(msg); return; }
    setError(null);

    // If advancing from Speech Analysis (step 2) to Assessment Report (step 3)
    if (step === 2) {
      setLoading(true);
      try {
        let age = 8;
        if (student && student.age) {
          age = parseInt(student.age, 10);
        }

        // Primary path: Phase 4 fusion predictor
        let res;
        try {
          res = await predictFusionPhase4(
            hwResult.features,
            spResult.features,
            parseInt(studentId),
            0.6,
            0.4,
          );
        } catch (fusionErr) {
          // Fallback path
          res = await predict({
            handwriting_features: hwResult.features,
            speech_features: spResult.features,
            student_id: parseInt(studentId),
          });
        }

        const merged = { ...res.data };
        const phasePredictions = {};
        
        try {
          const sp3 = await predictSpeechPhase3(spResult.features, parseInt(studentId));
          phasePredictions.phase3_speech = sp3.data?.prediction || null;
        } catch {
          phasePredictions.phase3_speech = null;
        }

        if (hwResult.imageFile) {
          try {
            const hw2 = await predictHandwritingPhase2(hwResult.imageFile, parseInt(studentId));
            phasePredictions.phase2_handwriting = hw2.data?.prediction || null;
          } catch {
            phasePredictions.phase2_handwriting = null;
          }
        }

        merged.phase_predictions = phasePredictions;
        setPrediction(merged);
      } catch (err) {
        setError(err.response?.data?.error || 'Prediction failed. Is the backend running?');
        setLoading(false);
        return;
      }
      setLoading(false);
    }
    setStep((s) => s + 1);
  };

  const handleBack = () => { setError(null); setStep((s) => s - 1); };

  const stepContent = () => {
    switch (step) {
      case 0: return <ConsentScreen accepted={consent} onAccept={setConsent} />;
      case 1: return <HandwritingTest onResult={setHwResult} result={hwResult} />;
      case 2: return <SpeechTest onResult={setSpResult} result={spResult} />;
      case 3: return (
        <AssessmentReport
          studentInfo={student}
          handwriting={hwResult}
          speech={spResult}
          prediction={prediction}
          onFinish={() => navigate(`/student/${studentId}`)}
        />
      );
      default: return null;
    }
  };

  if (initLoading) {
    return <Box mt={8} textAlign="center"><LinearProgress /></Box>;
  }

  if (!student && !initLoading) {
    return (
      <Container maxWidth="md" sx={{ mt: 8 }}>
        <Alert severity="error">Student not found. <Button onClick={() => navigate('/')}>Go Home</Button></Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom align="center">
        Screening Assessment: {student.name}
      </Typography>
      
      <StepIndicator steps={STEPS} active={step} />

      {loading && <LinearProgress sx={{ my: 2 }} />}
      {error && <Alert severity="error" sx={{ my: 2 }}>{error}</Alert>}

      <Paper elevation={2} sx={{ mt: 3, mb: 4, p: 3 }}>
        {stepContent()}
      </Paper>

      {/* Navigation */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="outlined"
          startIcon={<NavigateBefore />}
          disabled={step === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        {step < STEPS.length - 1 && (
          <Button
            variant="contained"
            endIcon={<NavigateNext />}
            onClick={handleNext}
            disabled={loading}
          >
            {step === 2 ? 'Generate Report' : 'Next'}
          </Button>
        )}
      </Box>

      {/* Footer */}
      <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary', fontSize: 12 }}>
        This tool is for screening purposes only and does not constitute a medical diagnosis.
      </Box>
    </Container>
  );
}
