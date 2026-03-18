import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Box, Container, CircularProgress, Alert, Button } from '@mui/material';
import { ArrowBack } from '@mui/icons-material';
import { getScreeningById } from '../utils/api';
import AssessmentReport from './AssessmentReport';

export default function ScreeningReportView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reportData, setReportData] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const res = await getScreeningById(id);
        const record = res.data;
        
        // Reconstruct props for AssessmentReport
        const stuInfo = {
          name: record.student.name,
          age: record.student.age,
          grade: record.student.grade,
          school: record.student.school,
          id: record.student.id,
          teacher: ''
        };
        
        const predObj = {
          prediction: {
            condition: record.predicted_condition,
            confidence: record.confidence,
            probabilities: record.probabilities
          },
          features: record.features || {},
          phase_predictions: record.phase_predictions || {},
          explanation: record.explanation || null,
          recommendations: record.recommendations || null
        };

        setReportData({ studentInfo: stuInfo, prediction: predObj });
      } catch (err) {
        console.error("Failed to fetch screening:", err);
        setError("Failed to load screening report. It may not exist or access is denied.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchReport();
  }, [id]);

  if (loading) {
    return <Box display="flex" justifyContent="center" mt={8}><CircularProgress /></Box>;
  }

  if (error || !reportData) {
    return (
      <Container maxWidth="md" sx={{ mt: 8 }}>
        <Alert severity="error">{error || "Report not found"}</Alert>
        <Button onClick={() => navigate(-1)} sx={{ mt: 2 }}>Go Back</Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Button 
        startIcon={<ArrowBack />} 
        onClick={() => navigate(`/student/${reportData.studentInfo.id}`)}
        sx={{ mb: 3 }}
      >
        Back to Student Dashboard
      </Button>
      <AssessmentReport 
        studentInfo={reportData.studentInfo}
        prediction={reportData.prediction}
        handwriting={null}
        speech={null}
      />
    </Container>
  );
}
