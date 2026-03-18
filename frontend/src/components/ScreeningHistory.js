import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, Chip, Button, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, Alert
} from '@mui/material';
import { Refresh, ArrowBack, Visibility, History } from '@mui/icons-material';
import { getScreenings } from '../utils/api';
import AssessmentReport from './AssessmentReport';

export default function ScreeningHistory({ onBack }) {
  const [screenings, setScreenings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // State for rendering a past report
  const [selectedReport, setSelectedReport] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getScreenings();
      setScreenings(res.data.screenings || []);
    } catch (err) {
      console.error("Failed to fetch history:", err);
      setError("Failed to load screening history. Is the backend running?");
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleRowClick = (record) => {
    // Reconstruct the props expected by AssessmentReport from the API record
    // Student Info
    const stuInfo = {
      name: record.student.name,
      age: record.student.age,
      grade: record.student.grade,
      school: record.student.school,
      id: record.student.id,
      teacher: ''
    };
    
    // Prediction Object
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

    // The backend might not explicitly store raw scores separate from prediction payload,
    // so we pass what we have
    setSelectedReport({ studentInfo: stuInfo, prediction: predObj });
  };

  if (selectedReport) {
    return (
      <Box>
        <Button 
          startIcon={<ArrowBack />} 
          onClick={() => setSelectedReport(null)}
          sx={{ mb: 2 }}
        >
          Back to History
        </Button>
        <AssessmentReport 
          studentInfo={selectedReport.studentInfo}
          prediction={selectedReport.prediction}
          handwriting={null} // We don't reconstruct full raw UI state, just the payload
          speech={null}
        />
      </Box>
    );
  }

  return (
    <Card elevation={2}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <History color="primary" />
            <Typography variant="h5">Screening History</Typography>
          </Box>
          <Box>
            <IconButton onClick={fetchHistory} disabled={loading} title="Refresh">
              <Refresh />
            </IconButton>
            <Button variant="outlined" startIcon={<ArrowBack />} onClick={onBack} sx={{ ml: 1 }}>
              Back to App
            </Button>
          </Box>
        </Box>

        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
            <CircularProgress />
          </Box>
        ) : screenings.length === 0 ? (
          <Alert severity="info" sx={{ py: 3 }}>
            No past screening results found in the database. Complete a new screening to see it here.
          </Alert>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table size="medium">
              <TableHead sx={{ bgcolor: 'background.default' }}>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Student</TableCell>
                  <TableCell>Result</TableCell>
                  <TableCell align="right">Confidence</TableCell>
                  <TableCell align="center">Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {screenings.map((row) => (
                  <TableRow key={row.id} hover sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                    <TableCell>{new Date(row.created_at).toLocaleString()}</TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>{row.student.name || 'Unknown'}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Age: {row.student.age} | Grade: {row.student.grade || 'N/A'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={row.predicted_condition?.replace(/_/g, ' ').toUpperCase() || 'UNKNOWN'}
                        color={row.predicted_condition === 'normal' ? 'success' : 'warning'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="right">
                      {(row.confidence * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="center">
                      <Button size="small" variant="text" onClick={() => handleRowClick(row)}>
                        View Report
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
}
