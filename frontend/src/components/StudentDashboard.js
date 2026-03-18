import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Container, 
  Paper, 
  Typography, 
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  CircularProgress,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import { 
  ArrowBack, 
  AddCircleOutline, 
  History, 
  Assessment 
} from '@mui/icons-material';
import { getStudentById } from '../utils/api';

const StudentDashboard = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const [student, setStudent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStudentDetails();
  }, [id]);

  const fetchStudentDetails = async () => {
    try {
      const res = await getStudentById(id);
      setStudent(res.data.student);
    } catch (err) {
      console.error('Failed to fetch student details', err);
    } finally {
      setLoading(false);
    }
  };

  const startNewAssessment = () => {
    navigate(`/wizard/${id}`);
  };

  if (loading) return <Box display="flex" justifyContent="center" mt={8}><CircularProgress /></Box>;
  if (!student) return <Container><Typography color="error" mt={8}>Student not found or access denied.</Typography></Container>;

  const latestScreening = student.screenings && student.screenings.length > 0 
    ? student.screenings[0] 
    : null;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Button 
        startIcon={<ArrowBack />} 
        onClick={() => navigate('/')}
        sx={{ mb: 2 }}
      >
        Back to Dashboard
      </Button>

      <Grid container spacing={4}>
        {/* Left Column: Profile & Action */}
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h4" gutterBottom>{student.name}</Typography>
            <Typography color="textSecondary" variant="subtitle1" gutterBottom>
              Age: {student.age} | Grade: {student.grade || 'N/A'}
            </Typography>
            <Typography color="textSecondary" variant="body2" gutterBottom>
              School: {student.school || 'N/A'}
            </Typography>
            
            <Box mt={4}>
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth 
                size="large"
                startIcon={<AddCircleOutline />}
                onClick={startNewAssessment}
              >
                Start New Assessment
              </Button>
            </Box>
          </Paper>

          {/* Mini-Summary of Latest Screening */}
          {latestScreening && (
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Latest Status
                </Typography>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  {new Date(latestScreening.created_at).toLocaleDateString()}
                </Typography>
                <Chip 
                  label={latestScreening.predicted_condition} 
                  color={
                    latestScreening.predicted_condition === 'Normal' ? 'success' : 
                    latestScreening.predicted_condition === 'Dyslexia' ? 'warning' : 'error'
                  }
                  sx={{ mt: 1, mb: 2, fontSize: '1.1rem', py: 2 }}
                />
                <Typography variant="body2">
                  Confidence: {(latestScreening.confidence * 100).toFixed(1)}%
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => navigate(`/screening/${latestScreening.id}`)}>
                  View Full Report
                </Button>
              </CardActions>
            </Card>
          )}
        </Grid>

        {/* Right Column: History */}
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" component="h2" gutterBottom display="flex" alignItems="center">
              <History sx={{ mr: 1 }} /> Screening History
            </Typography>
            <Divider sx={{ mb: 2 }} />

            {student.screenings && student.screenings.length > 0 ? (
              <List>
                {student.screenings.map((screening) => (
                  <ListItem 
                    key={screening.id} 
                    divider
                    button
                    onClick={() => navigate(`/screening/${screening.id}`)}
                  >
                    <ListItemIcon>
                      <Assessment color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="subtitle1" fontWeight="bold">
                            {screening.predicted_condition}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            {new Date(screening.created_at).toLocaleDateString()}
                          </Typography>
                        </Box>
                      }
                      secondary={`Confidence: ${(screening.confidence * 100).toFixed(1)}% | Severity: ${screening.severity_level}`}
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box py={4} textAlign="center">
                <Typography color="textSecondary" gutterBottom>
                  No screenings recorded yet.
                </Typography>
                <Button variant="outlined" startIcon={<AddCircleOutline />} onClick={startNewAssessment} sx={{ mt: 1 }}>
                  Conduct First Assessment
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default StudentDashboard;
