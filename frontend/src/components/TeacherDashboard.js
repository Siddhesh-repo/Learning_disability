import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress
} from '@mui/material';
import { PersonAdd, ChevronRight } from '@mui/icons-material';
import { getStudents, createStudent } from '../utils/api';

const TeacherDashboard = () => {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openDialog, setOpenDialog] = useState(false);
  const [newStudent, setNewStudent] = useState({ name: '', age: 8, grade: '', school: '' });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchStudents();
  }, []);

  const fetchStudents = async () => {
    try {
      const res = await getStudents();
      setStudents(res.data.students || []);
    } catch (err) {
      console.error('Failed to fetch students', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateStudent = async () => {
    try {
      await createStudent(newStudent);
      setOpenDialog(false);
      setNewStudent({ name: '', age: 8, grade: '', school: '' });
      fetchStudents();
    } catch (err) {
      console.error('Failed to create student', err);
      alert('Failed to create student record.');
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4" component="h1">
          My Students
        </Typography>
        <Button 
          variant="contained" 
          startIcon={<PersonAdd />}
          onClick={() => setOpenDialog(true)}
        >
          Add New Student
        </Button>
      </Box>

      {loading ? (
        <Box display="flex" justifyContent="center"><CircularProgress /></Box>
      ) : students.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="textSecondary" gutterBottom>
            You haven't added any students yet.
          </Typography>
          <Button variant="outlined" sx={{ mt: 2 }} onClick={() => setOpenDialog(true)}>
            Add your first student
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={4}>
          {students.map(student => (
            <Grid item key={student.id} xs={12} sm={6} md={4}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h5" component="h2" gutterBottom>
                    {student.name}
                  </Typography>
                  <Typography color="textSecondary">
                    Age: {student.age} | Grade: {student.grade || 'N/A'}
                  </Typography>
                  <Typography color="textSecondary" variant="body2">
                    School: {student.school || 'N/A'}
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button 
                    size="small" 
                    color="primary" 
                    fullWidth 
                    endIcon={<ChevronRight />}
                    onClick={() => navigate(`/student/${student.id}`)}
                  >
                    View Profile & History
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Add Student Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
        <DialogTitle>Add New Student</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Full Name"
            type="text"
            fullWidth
            required
            value={newStudent.name}
            onChange={(e) => setNewStudent({...newStudent, name: e.target.value})}
          />
          <TextField
            margin="dense"
            label="Age"
            type="number"
            fullWidth
            value={newStudent.age}
            onChange={(e) => setNewStudent({...newStudent, age: e.target.value})}
          />
          <TextField
            margin="dense"
            label="Grade"
            type="text"
            fullWidth
            value={newStudent.grade}
            onChange={(e) => setNewStudent({...newStudent, grade: e.target.value})}
          />
          <TextField
            margin="dense"
            label="School"
            type="text"
            fullWidth
            value={newStudent.school}
            onChange={(e) => setNewStudent({...newStudent, school: e.target.value})}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateStudent} variant="contained" disabled={!newStudent.name}>
            Add Student
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default TeacherDashboard;
