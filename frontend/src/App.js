import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, AppBar, Toolbar, Typography, Button } from '@mui/material';
import { School, MeetingRoom } from '@mui/icons-material';

import './App.css';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';

import Login from './components/Login';
import Register from './components/Register';
import TeacherDashboard from './components/TeacherDashboard';
import StudentDashboard from './components/StudentDashboard';
import ScreeningWizard from './components/ScreeningWizard';
import ScreeningReportView from './components/ScreeningReportView';

const theme = createTheme({
  palette: {
    primary: { main: '#1565c0' },
    secondary: { main: '#e65100' },
    background: { default: '#f5f7fa' },
  },
  typography: {
    fontFamily: '"Roboto", "Segoe UI", sans-serif',
  },
});

const NavigationBar = () => {
  const { token, logout, user } = useAuth();
  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <School sx={{ mr: 1.5 }} />
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Learning Disability Screening System
        </Typography>
        {token && (
          <Button 
            color="inherit" 
            onClick={logout}
            startIcon={<MeetingRoom />}
          >
            Logout {user?.full_name ? `(${user.full_name})` : ''}
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <NavigationBar />
            
            <Box sx={{ flexGrow: 1 }}>
              <Routes>
                {/* Public / Auth Routes */}
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                
                {/* Protected Routes */}
                <Route 
                  path="/" 
                  element={<ProtectedRoute><TeacherDashboard /></ProtectedRoute>} 
                />
                <Route 
                  path="/student/:id" 
                  element={<ProtectedRoute><StudentDashboard /></ProtectedRoute>} 
                />
                <Route 
                  path="/wizard/:studentId" 
                  element={<ProtectedRoute><ScreeningWizard /></ProtectedRoute>} 
                />
                <Route 
                  path="/screening/:id" 
                  element={<ProtectedRoute><ScreeningReportView /></ProtectedRoute>} 
                />
                
                {/* Fallback */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
